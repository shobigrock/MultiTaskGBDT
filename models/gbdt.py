"""
マルチタスク勾配ブースティング決定木（MT-GBDT）のスクラッチ実装

このモジュールは、論文「MT-GBM: A multi-task Gradient Boosting Machine with shared decision trees」
に基づいたマルチタスク勾配ブースティング決定木のスクラッチ実装を提供します。
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import time
from .base import MTGBMBase


class DecisionTreeNode:
    """
    決定木のノードクラス
    
    Attributes:
    -----------
    feature_idx : int or None
        分割に使用する特徴のインデックス（リーフノードの場合はNone）
    threshold : float or None
        分割の閾値（リーフノードの場合はNone）
    left : DecisionTreeNode or None
        左の子ノード
    right : DecisionTreeNode or None
        右の子ノード
    is_leaf : bool
        リーフノードかどうか
    values : array-like, shape=(n_tasks,)
        リーフノードの場合の予測値（各タスクごと）
    """
    
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.values = None
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        ノードを通じて予測を行う
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        predictions : array-like, shape=(n_samples, n_tasks)
            各サンプルの各タスクの予測値
        """
        if self.is_leaf:
            return np.tile(self.values, (X.shape[0], 1))
        
        left_mask = X[:, self.feature_idx] <= self.threshold
        right_mask = ~left_mask
        
        predictions = np.zeros((X.shape[0], self.values.shape[0]))
        
        if np.any(left_mask):
            predictions[left_mask] = self.left.predict(X[left_mask])
        
        if np.any(right_mask):
            predictions[right_mask] = self.right.predict(X[right_mask])
            
        return predictions


class MultiTaskDecisionTree:
    """
    マルチタスク決定木クラス
    
    Attributes:
    -----------
    max_depth : int
        木の最大深さ
    min_samples_split : int
        分割に必要な最小サンプル数
    min_samples_leaf : int
        リーフノードに必要な最小サンプル数
    n_tasks : int
        タスク数
    root : DecisionTreeNode
        ルートノード
    """
    
    def __init__(self, 
                 max_depth: int = 3, 
                 min_samples_split: int = 2, 
                 min_samples_leaf: int = 1,
                 weighting_strategy: str = "mtgbm",
                 lambda_weight: float = 0.9,
                 gain_threshold: float = 0.1,
                 track_split_gains: bool = True,
                 random_state: Optional[int] = None):
        """
        初期化メソッド
        
        Parameters:
        -----------
        max_depth : int, default=3
            木の最大深さ
        min_samples_split : int, default=2
            分割に必要な最小サンプル数
        min_samples_leaf : int, default=1
            リーフノードに必要な最小サンプル数
        weighting_strategy : str, default="mtgbm"
            重み戦略
        lambda_weight : float, default=0.9
            λパラメータ
        gain_threshold : float, default=0.1
            情報利得閾値
        track_split_gains : bool, default=True
            分割利得の記録有無
        random_state : int, optional
            乱数シード
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.weighting_strategy = weighting_strategy
        self.lambda_weight = lambda_weight
        self.gain_threshold = gain_threshold
        self.track_split_gains = track_split_gains
        self.root = None
        
        # 分割利得の記録
        self.split_gains = []
        self.current_iteration = 0
        
        # 提案戦略用の現在の重み
        if weighting_strategy == "proposed":
            self._current_weights = np.array([lambda_weight, 1 - lambda_weight])
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, 
            X: np.ndarray, 
            gradients: np.ndarray, 
            hessians: np.ndarray, 
            ensemble_gradients: Optional[np.ndarray] = None,
            ensemble_hessians: Optional[np.ndarray] = None,
            ensemble_weights: Optional[np.ndarray] = None,
            task_correlations: Optional[np.ndarray] = None) -> 'MultiTaskDecisionTree':
        """
        マルチタスク決定木を学習
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのヘシアン
        ensemble_gradients : array-like, shape=(n_samples,), optional
            加重和された勾配（分割に使用）
        ensemble_hessians : array-like, shape=(n_samples,), optional
            加重和されたヘシアン（分割に使用）
        ensemble_weights : array-like, shape=(n_tasks,), optional
            各タスクのアンサンブル重み
        task_correlations : array-like, shape=(n_tasks, n_tasks), optional
            タスク間の相関行列
            
        Returns:
        --------
        self : MultiTaskDecisionTree
            学習済みモデル
        """
        self.n_tasks = gradients.shape[1]
        
        # アンサンブル重みが指定されていない場合は均等に設定
        if ensemble_weights is None:
            ensemble_weights = np.ones(self.n_tasks) / self.n_tasks
        
        # タスク相関が指定されていない場合は単位行列を使用
        if task_correlations is None:
            task_correlations = np.eye(self.n_tasks)
        
        # 加重和された勾配・ヘシアンが指定されていない場合は計算
        if ensemble_gradients is None or ensemble_hessians is None:
            ensemble_gradients = np.mean(gradients, axis=1)
            ensemble_hessians = np.mean(hessians, axis=1)
        
        # ルートノードを作成し、再帰的に木を構築
        self.root = DecisionTreeNode()
        self._build_tree(
            self.root, X, gradients, hessians, ensemble_gradients, ensemble_hessians, 
            ensemble_weights, task_correlations, depth=0
        )
        
        return self
    
    def _compute_ensemble_gradients_hessians(self, 
                                            gradients: np.ndarray, 
                                            hessians: np.ndarray, 
                                            ensemble_weights: np.ndarray,
                                            task_correlations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        アンサンブル勾配とヘシアンを計算
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのヘシアン
        ensemble_weights : array-like, shape=(n_tasks,)
            各タスクのアンサンブル重み
        task_correlations : array-like, shape=(n_tasks, n_tasks)
            タスク間の相関行列
            
        Returns:
        --------
        ensemble_gradients : array-like, shape=(n_samples,)
            各サンプルのアンサンブル勾配
        ensemble_hessians : array-like, shape=(n_samples,)
            各サンプルのアンサンブルヘシアン
        """
        # 論文のアルゴリズム2に基づいたアンサンブル勾配の計算
        # 各タスクの勾配に重みを掛けて合計
        ensemble_gradients = np.zeros(gradients.shape[0])
        ensemble_hessians = np.zeros(hessians.shape[0])
        
        # ランダムに1つまたは複数のタスクを強調
        # 論文では、γパラメータ（10〜100）を使用して特定のタスクを強調
        gamma = 50.0  # 論文の推奨値
        
        # 各タスクの重みを正規化（平均0.05、標準偏差0.01）
        normalized_weights = self._normalize_weights(ensemble_weights, mean=0.05, std=0.01)
        
        # ランダムに1つのタスクを選択して強調
        # 実際のアプリケーションでは、メインタスクを常に選択することも可能
        emphasized_task = np.random.choice(self.n_tasks)
        normalized_weights[emphasized_task] *= gamma
        
        # アンサンブル勾配を計算
        for i in range(self.n_tasks):
            ensemble_gradients += normalized_weights[i] * gradients[:, i]
        
        # ヘシアンの重みを正規化（平均1.0、標準偏差0.1）
        normalized_hessian_weights = self._normalize_weights(ensemble_weights, mean=1.0, std=0.1)
        
        # アンサンブルヘシアンを計算
        for i in range(self.n_tasks):
            ensemble_hessians += normalized_hessian_weights[i] * hessians[:, i]
        
        return ensemble_gradients, ensemble_hessians
    
    def _normalize_weights(self, weights: np.ndarray, mean: float = 0.05, std: float = 0.01) -> np.ndarray:
        """
        重みを指定された平均と標準偏差に正規化
        
        Parameters:
        -----------
        weights : array-like
            正規化する重み
        mean : float, default=0.05
            目標平均値
        std : float, default=0.01
            目標標準偏差
            
        Returns:
        --------
        normalized_weights : array-like
            正規化された重み
        """
        # 現在の平均と標準偏差を計算
        current_mean = np.mean(weights)
        current_std = np.std(weights)
        
        if current_std == 0:
            # 全ての重みが同じ場合
            return np.ones_like(weights) * mean
        
        # 正規化
        normalized = (weights - current_mean) / current_std * std + mean
        
        # 負の値を防止
        normalized = np.maximum(normalized, 1e-6)
        
        return normalized
    
    def _build_tree(self, 
                   node: DecisionTreeNode, 
                   X: np.ndarray, 
                   gradients: np.ndarray, 
                   hessians: np.ndarray,
                   ensemble_gradients: np.ndarray, 
                   ensemble_hessians: np.ndarray,
                   ensemble_weights: np.ndarray,
                   task_correlations: np.ndarray,
                   depth: int) -> None:
        """
        再帰的に決定木を構築
        
        Parameters:
        -----------
        node : DecisionTreeNode
            現在のノード
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのヘシアン
        ensemble_gradients : array-like, shape=(n_samples,)
            各サンプルのアンサンブル勾配
        ensemble_hessians : array-like, shape=(n_samples,)
            各サンプルのアンサンブルヘシアン
        ensemble_weights : array-like, shape=(n_tasks,)
            各タスクのアンサンブル重み
        task_correlations : array-like, shape=(n_tasks, n_tasks)
            タスク間の相関行列
        depth : int
            現在の深さ
        """
        n_samples = X.shape[0]
        
        # リーフノードの場合の値を計算
        node.values = self._compute_leaf_values(gradients, hessians, task_correlations)
        
        # 停止条件をチェック
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_samples < 2 * self.min_samples_leaf):
            node.is_leaf = True
            return
        
        # 最適な分割を見つける
        best_gain = 0.0
        best_feature_idx = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None
        
        # 各特徴について最適な分割を探索
        for feature_idx in range(X.shape[1]):
            # ソートされた特徴値とそれに対応するインデックス
            sorted_indices = np.argsort(X[:, feature_idx])
            sorted_feature = X[sorted_indices, feature_idx]
            sorted_ensemble_gradients = ensemble_gradients[sorted_indices]
            sorted_ensemble_hessians = ensemble_hessians[sorted_indices]
            
            # 累積和を計算
            left_sum_gradients = np.cumsum(sorted_ensemble_gradients)
            left_sum_hessians = np.cumsum(sorted_ensemble_hessians)
            
            right_sum_gradients = left_sum_gradients[-1] - left_sum_gradients
            right_sum_hessians = left_sum_hessians[-1] - left_sum_hessians
            
            # 各分割点での情報利得を計算
            for i in range(1, n_samples):
                # 連続した値が同じ場合はスキップ
                if sorted_feature[i] == sorted_feature[i-1]:
                    continue
                
                # 左右の子ノードのサンプル数をチェック
                if i < self.min_samples_leaf or n_samples - i < self.min_samples_leaf:
                    continue
                
                # 分割の閾値
                threshold = (sorted_feature[i-1] + sorted_feature[i]) / 2
                
                # 左右の子ノードの勾配とヘシアンの合計（加重和された値を使用）
                left_gradient = left_sum_gradients[i-1]
                left_hessian = left_sum_hessians[i-1]
                right_gradient = right_sum_gradients[i-1]
                right_hessian = right_sum_hessians[i-1]
                
                # 正則化項
                lambda_reg = 1.0  # L2正則化パラメータ
                gamma_reg = 0.0   # 複雑性に対するペナルティ
                
                # 情報利得を計算（XGBoostの公式を使用）
                if left_hessian > 0 and right_hessian > 0 and (left_hessian + right_hessian) > 0:
                    gain = 0.5 * (
                        (left_gradient**2) / (left_hessian + lambda_reg) +
                        (right_gradient**2) / (right_hessian + lambda_reg) -
                        ((left_gradient + right_gradient)**2) / (left_hessian + right_hessian + lambda_reg)
                    ) - gamma_reg
                else:
                    gain = 0.0
                
                # より良い分割が見つかった場合は更新
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    best_left_indices = sorted_indices[:i]
                    best_right_indices = sorted_indices[i:]
        
        # 有効な分割が見つからなかった場合
        if best_gain <= 0 or best_feature_idx is None:
            node.is_leaf = True
            return
        
        # ノードに分割情報を設定
        node.feature_idx = best_feature_idx
        node.threshold = best_threshold
        
        # 左右の子ノードを作成
        node.left = DecisionTreeNode()
        node.right = DecisionTreeNode()
        
        # 左右の子ノードを再帰的に構築
        self._build_tree(
            node.left, 
            X[best_left_indices], 
            gradients[best_left_indices], 
            hessians[best_left_indices],
            ensemble_gradients[best_left_indices], 
            ensemble_hessians[best_left_indices],
            ensemble_weights,
            task_correlations,
            depth + 1
        )
        
        self._build_tree(
            node.right, 
            X[best_right_indices], 
            gradients[best_right_indices], 
            hessians[best_right_indices],
            ensemble_gradients[best_right_indices], 
            ensemble_hessians[best_right_indices],
            ensemble_weights,
            task_correlations,
            depth + 1
        )
    
    def _compute_leaf_values(self, 
                            gradients: np.ndarray, 
                            hessians: np.ndarray,
                            task_correlations: np.ndarray) -> np.ndarray:
        """
        リーフノードの値を計算
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのヘシアン
        task_correlations : array-like, shape=(n_tasks, n_tasks)
            タスク間の相関行列
            
        Returns:
        --------
        leaf_values : array-like, shape=(n_tasks,)
            各タスクのリーフ値
        """
        n_samples = gradients.shape[0]
        n_tasks = gradients.shape[1]
        
        # 各タスクの勾配とヘシアンの合計
        sum_gradients = np.sum(gradients, axis=0)
        sum_hessians = np.sum(hessians, axis=0)
        
        # L2正則化パラメータ
        lambda_reg = 1.0
        
        # 各タスクのリーフ値を計算（標準的なXGBoostの式）
        leaf_values = np.zeros(n_tasks)
        for i in range(n_tasks):
            if sum_hessians[i] > 0:
                # XGBoostの標準的な葉値計算: w* = -G / (H + λ)
                leaf_values[i] = -sum_gradients[i] / (sum_hessians[i] + lambda_reg)
            else:
                leaf_values[i] = 0.0
        
        return leaf_values
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        学習済みモデルで予測
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        predictions : array-like, shape=(n_samples, n_tasks)
            各サンプルの各タスクの予測値
        """
        if self.root is None:
            raise ValueError("Model has not been trained yet")
        
        return self.root.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        確率予測（二値分類用）
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        y_proba : array-like, shape=(n_samples, n_tasks)
            各タスクの確率予測値（0-1の範囲）
        """
        if self.loss == "mse":
            # MSE損失の場合は通常の予測値をクリッピング
            y_pred = self.predict(X)
            return np.clip(y_pred, 0, 1)
        elif self.loss == "logloss":
            # logloss損失の場合は既に確率になっている
            return self.predict(X)
        else:
            raise ValueError(f"Unsupported loss function for probability prediction: {self.loss}")
    
    def predict_binary(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        二値予測（0/1の分類結果）
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        threshold : float, default=0.5
            分類の閾値
            
        Returns:
        --------
        y_binary : array-like, shape=(n_samples, n_tasks)
            各タスクの二値予測結果（0 or 1）
        """
        y_proba = self.predict_proba(X)
        return (y_proba >= threshold).astype(int)


class MTGBDT(MTGBMBase):
    """
    マルチタスク勾配ブースティング決定木（MT-GBDT）のスクラッチ実装
    
    Attributes:
    -----------
    n_estimators : int
        ブースティング反復回数（木の数）
    learning_rate : float
        学習率（各木の寄与度）
    max_depth : int
        各木の最大深さ
    min_samples_split : int
        分割に必要な最小サンプル数
    min_samples_leaf : int
        リーフノードに必要な最小サンプル数
    subsample : float
        各木を構築する際のサンプリング率
    colsample_bytree : float
        各木を構築する際の特徴サンプリング率
    n_tasks : int
        タスク数
    trees : list of MultiTaskDecisionTree
        学習済みの木のリスト
    initial_predictions : array-like, shape=(n_tasks,)
        初期予測値
    """
    
    def __init__(self, 
        n_estimators: int = 100, 
        learning_rate: float = 0.1, 
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        gradient_weights: Optional[np.ndarray] = None,
        normalize_gradients: bool = False,
        loss: str = "mse",
        weighting_strategy: str = "mtgbm",
        lambda_weight: float = 0.9,
        gain_threshold: float = 0.1,
        track_split_gains: bool = True,
        random_state: Optional[int] = None):
        """
        初期化メソッド
        
        Parameters:
        -----------
        n_estimators : int, default=100
            ブースティング反復回数（木の数）
        learning_rate : float, default=0.1
            学習率（各木の寄与度）
        max_depth : int, default=3
            各木の最大深さ
        min_samples_split : int, default=2
            分割に必要な最小サンプル数
        min_samples_leaf : int, default=1
            リーフノードに必要な最小サンプル数
        subsample : float, default=1.0
            各木を構築する際のサンプリング率
        colsample_bytree : float, default=1.0
            各木を構築する際の特徴サンプリング率
        gradient_weights : array-like, optional
            勾配・ヘシアンの加重和用の重み（W パラメータ）。
            Noneの場合は[50, 1]をデフォルトとし、50%の確率で[1, 50]に反転
        normalize_gradients : bool, default=False
            勾配・ヘシアンを標準化するかどうか
        loss : str, default="mse"
            損失関数の種類（"mse": 平均二乗誤差, "logloss": クロスエントロピー損失）
        weighting_strategy : str, default="mtgbm"
            重み戦略（"mtgbm": 従来手法, "proposed": 提案手法）
        lambda_weight : float, default=0.9
            タスク重みのλパラメータ（0.5-1.0）
        gain_threshold : float, default=0.1
            情報利得の閾値δ（提案手法で使用）
        track_split_gains : bool, default=True
            分割利得の履歴を記録するかどうか
        random_state : int, optional
            乱数シード
        """
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gradient_weights = gradient_weights
        self.normalize_gradients = normalize_gradients
        self.loss = loss
        self.weighting_strategy = weighting_strategy
        self.lambda_weight = lambda_weight
        self.gain_threshold = gain_threshold
        self.track_split_gains = track_split_gains
        
        # 新しいパラメータの検証
        if self.weighting_strategy not in ["mtgbm", "proposed"]:
            raise ValueError(f"Unsupported weighting strategy: {self.weighting_strategy}. Use 'mtgbm' or 'proposed'.")
        
        if not 0.5 <= self.lambda_weight <= 1.0:
            raise ValueError("lambda_weight must be between 0.5 and 1.0")
        
        # 履歴記録の初期化
        if self.track_split_gains:
            self.split_gains_history = []
            self.iteration_gains = []
            self.weight_switches = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_initial_predictions(self, y_multi: np.ndarray) -> np.ndarray:
        """
        初期予測値を計算
        
        Parameters:
        -----------
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクのターゲット値
            
        Returns:
        --------
        initial_predictions : array-like, shape=(n_tasks,)
            各タスクの初期予測値
        """
        if self.loss == "mse":
            # 回帰の場合：各タスクの平均値
            return np.mean(y_multi, axis=0)
        elif self.loss == "logloss":
            # 二値分類の場合：各タスクのlogit(平均確率)
            mean_probs = np.mean(y_multi, axis=0)
            # 0や1を避けるためのクリッピング
            mean_probs = np.clip(mean_probs, 1e-7, 1 - 1e-7)
            # logit変換
            return np.log(mean_probs / (1 - mean_probs))
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")
    
    def _compute_gradients_hessians(self, y_multi: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        勾配とヘシアンを計算
        
        Parameters:
        -----------
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクのターゲット値
        y_pred : array-like, shape=(n_samples, n_tasks)
            現在の予測値
            
        Returns:
        --------
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのヘシアン
        """
        if self.loss == "mse":
            # 二乗誤差損失の勾配とヘシアン
            # 勾配: 2 * (予測値 - 真の値)
            # ヘシアン: 2
            gradients = 2 * (y_pred - y_multi)
            hessians = np.ones_like(gradients) * 2
            
        elif self.loss == "logloss":
            # クロスエントロピー損失の勾配とヘシアン（二値分類）
            # シグモイド関数で確率に変換
            probs = 1 / (1 + np.exp(-y_pred))  # sigmoid(y_pred)
            # 数値安定性のためクリッピング
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            
            # 勾配: p - y (where p = sigmoid(f(x)))
            gradients = probs - y_multi
            
            # ヘシアン: p * (1 - p)
            hessians = probs * (1 - probs)
            # 数値安定性のため最小値を設定
            hessians = np.maximum(hessians, 1e-7)
            
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")
        
        return gradients, hessians
    
    def _compute_task_correlations(self, gradients: np.ndarray) -> np.ndarray:
        """
        タスク間の相関行列を計算
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配
            
        Returns:
        --------
        task_correlations : array-like, shape=(n_tasks, n_tasks)
            タスク間の相関行列
        """
        # 勾配ベクトル間の相関を計算
        corr = np.corrcoef(gradients.T)
        
        # NaNを0に置き換え
        corr = np.nan_to_num(corr)
        
        return corr
    
    def _normalize_gradients_hessians(self, gradients: np.ndarray, hessians: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        勾配とヘシアンを標準化（平均値・標本分散による正規化）
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのヘシアン
            
        Returns:
        --------
        normalized_gradients : array-like, shape=(n_samples, n_tasks)
            標準化された勾配
        normalized_hessians : array-like, shape=(n_samples, n_tasks)
            標準化されたヘシアン
        """
        n_samples, n_tasks = gradients.shape
        normalized_gradients = np.zeros_like(gradients)
        normalized_hessians = np.zeros_like(hessians)
        
        # 各タスクごとに標準化
        for task_idx in range(n_tasks):
            # 勾配の標準化
            grad_mean = np.mean(gradients[:, task_idx])
            grad_var = np.var(gradients[:, task_idx], ddof=1)  # 標本分散（不偏分散）
            
            if grad_var > 1e-8:  # ゼロ除算を防ぐ
                normalized_gradients[:, task_idx] = (gradients[:, task_idx] - grad_mean) / np.sqrt(grad_var)
            else:
                normalized_gradients[:, task_idx] = gradients[:, task_idx] - grad_mean
            
            # ヘシアンの標準化（損失関数に応じて処理を変更）
            if self.loss == "mse":
                # MSE損失の場合、ヘシアンは定数なので勾配の分散に基づいてスケール
                if grad_var > 1e-8:
                    normalized_hessians[:, task_idx] = hessians[:, task_idx] / np.sqrt(grad_var)
                else:
                    normalized_hessians[:, task_idx] = hessians[:, task_idx]
            elif self.loss == "logloss":
                # logloss損失の場合、ヘシアンも標準化
                hess_mean = np.mean(hessians[:, task_idx])
                hess_var = np.var(hessians[:, task_idx], ddof=1)
                
                if hess_var > 1e-8:
                    normalized_hessians[:, task_idx] = (hessians[:, task_idx] - hess_mean) / np.sqrt(hess_var)
                else:
                    normalized_hessians[:, task_idx] = hessians[:, task_idx] - hess_mean
        
        return normalized_gradients, normalized_hessians
    
    def _compute_weighted_ensemble_gradients_hessians(self, 
                                                     gradients: np.ndarray, 
                                                     hessians: np.ndarray,
                                                     weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        勾配とヘシアンの加重和を計算
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配（標準化済み）
        hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのヘシアン（標準化済み）
        weights : array-like, shape=(n_tasks,), optional
            各タスクの重み（W パラメータ）。Noneの場合は動的重みを使用
            
        Returns:
        --------
        ensemble_gradients : array-like, shape=(n_samples,)
            加重和された勾配
        ensemble_hessians : array-like, shape=(n_samples,)
            加重和されたヘシアン
        """
        n_samples, n_tasks = gradients.shape
        
        # 重みが指定されていない場合は動的重みを生成
        if weights is None:
            weights = self._get_dynamic_gradient_weights(n_tasks)
        else:
            # 重みを正規化
            weights = np.array(weights)
            weights = weights / np.sum(weights)
        
        # 加重和を計算
        ensemble_gradients = np.zeros(n_samples)
        ensemble_hessians = np.zeros(n_samples)
        
        for task_idx in range(n_tasks):
            ensemble_gradients += weights[task_idx] * gradients[:, task_idx]
            ensemble_hessians += weights[task_idx] * hessians[:, task_idx]
        
        return ensemble_gradients, ensemble_hessians
    
    def fit(self, X: np.ndarray, y_multi: np.ndarray, **kwargs) -> 'MTGBDT':
        """
        マルチタスクデータでモデルを学習
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクのターゲット値
        **kwargs : dict
            追加のパラメータ
            
        Returns:
        --------
        self : MTGBDT
            学習済みモデル
        """
        # 入力検証
        X, y_multi = self._validate_input(X, y_multi)
        
        # 初期予測値を計算
        self.initial_predictions = self._compute_initial_predictions(y_multi)
        
        # 現在の予測値を初期化
        y_pred = np.tile(self.initial_predictions, (X.shape[0], 1))
        
        # 学習開始時間
        start_time = time.time()
        
        # 各反復で木を構築
        for i in range(self.n_estimators):
            # 勾配とヘシアンを計算
            gradients, hessians = self._compute_gradients_hessians(y_multi, y_pred)
            
            # 勾配とヘシアンの標準化（オプション）
            if self.normalize_gradients:
                gradients, hessians = self._normalize_gradients_hessians(gradients, hessians)
            
            # 木に戦略パラメータを渡して構築
            tree = MultiTaskDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                weighting_strategy=self.weighting_strategy,
                lambda_weight=self.lambda_weight,
                gain_threshold=self.gain_threshold,
                track_gains=self.track_split_gains,
                random_state=self.random_state
            )
            
            tree.fit(X, gradients, hessians)
            
            # サブサンプリング
            if self.subsample < 1.0:
                sample_indices = np.random.choice(
                    X.shape[0], 
                    size=int(X.shape[0] * self.subsample), 
                    replace=False
                )
                X_subset = X[sample_indices]
                gradients_subset = gradients[sample_indices]
                hessians_subset = hessians[sample_indices]
                ensemble_gradients_subset = ensemble_gradients[sample_indices]
                ensemble_hessians_subset = ensemble_hessians[sample_indices]
            else:
                X_subset = X
                gradients_subset = gradients
                hessians_subset = hessians
                ensemble_gradients_subset = ensemble_gradients
                ensemble_hessians_subset = ensemble_hessians
            
            # 特徴量のサブサンプリング
            feature_indices = None
            if self.colsample_bytree < 1.0:
                n_features = X.shape[1]
                feature_indices = np.random.choice(
                    n_features, 
                    size=int(n_features * self.colsample_bytree), 
                    replace=False
                )
                X_subset = X_subset[:, feature_indices]
            
            # 新しい木を構築（加重和された勾配・ヘシアンを使用）
            tree = MultiTaskDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            
            # 木を学習（元の勾配・ヘシアンと加重和されたものを両方渡す）
            tree.fit(
                X_subset, 
                gradients_subset, 
                hessians_subset, 
                ensemble_gradients=ensemble_gradients_subset,
                ensemble_hessians=ensemble_hessians_subset,
                task_correlations=task_correlations
            )
            
            # 木を保存
            self.trees.append({
                'tree': tree,
                'feature_indices': feature_indices  # 特徴量サブサンプリングの情報を保存
            })
            
            # 予測値を更新
            if feature_indices is not None:
                # 特徴量サブサンプリングした場合
                X_for_update = X[:, feature_indices]
            else:
                X_for_update = X
            
            update = tree.predict(X_for_update)
            y_pred += self.learning_rate * update
            
            # 進捗を表示（オプション）
            if (i + 1) % 10 == 0 or i == 0 or i == self.n_estimators - 1:
                elapsed_time = time.time() - start_time
                mse = np.mean((y_multi - y_pred) ** 2)
                print(f"Iteration {i+1}/{self.n_estimators}, MSE: {mse:.6f}, Time: {elapsed_time:.2f}s")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        学習済みモデルで予測
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        y_pred : array-like, shape=(n_samples, n_tasks)
            各タスクの予測値
        """
        # 入力検証
        X, _ = self._validate_input(X)
        
        if not self.trees:
            raise ValueError("Model has not been trained yet")
        
        # 初期予測値
        y_pred = np.tile(self.initial_predictions, (X.shape[0], 1))
        
        # 各木の予測を加算
        for tree_info in self.trees:
            if isinstance(tree_info, dict):
                # 新しい形式（特徴量インデックス情報付き）
                tree = tree_info['tree']
                feature_indices = tree_info['feature_indices']
                
                if feature_indices is not None:
                    # 特徴量サブサンプリングした場合
                    X_subset = X[:, feature_indices]
                    tree_prediction = tree.predict(X_subset)
                else:
                    tree_prediction = tree.predict(X)
            else:
                # 古い形式（後方互換性）
                tree = tree_info
                tree_prediction = tree.predict(X)
            
            y_pred += self.learning_rate * tree_prediction
        
        # 損失関数に応じた後処理
        if self.loss == "logloss":
            # ロジスティック回帰の場合：シグモイド関数で確率に変換
            y_pred = 1 / (1 + np.exp(-y_pred))  # sigmoid(y_pred)
            # 数値安定性のためクリッピング
            y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        return y_pred
    
    def get_feature_importance(self) -> np.ndarray:
        """
        特徴量の重要度を計算
        
        Returns:
        --------
        feature_importance : array-like
            各特徴量の重要度
        """
        if not self.trees:
            raise ValueError("Model has not been trained yet")
        
        # 特徴量の数を取得
        n_features = max([tree.root.feature_idx for tree in self.trees if tree.root.feature_idx is not None]) + 1
        
        # 特徴量の重要度を初期化
        feature_importance = np.zeros(n_features)
        
        # 各木の各ノードで使用される特徴量をカウント
        for tree in self.trees:
            self._count_feature_usage(tree.root, feature_importance)
        
        # 合計で正規化
        if np.sum(feature_importance) > 0:
            feature_importance = feature_importance / np.sum(feature_importance)
        
        return feature_importance
    
    def _count_feature_usage(self, node: DecisionTreeNode, feature_importance: np.ndarray) -> None:
        """
        ノードで使用される特徴量をカウント
        
        Parameters:
        -----------
        node : DecisionTreeNode
            カウントするノード
        feature_importance : array-like
            特徴量の重要度を格納する配列
        """
        if node is None or node.is_leaf:
            return
        
        # 特徴量の使用をカウント
        feature_importance[node.feature_idx] += 1
        
        # 子ノードを再帰的に処理
        self._count_feature_usage(node.left, feature_importance)
        self._count_feature_usage(node.right, feature_importance)
    
    def _get_dynamic_gradient_weights(self, n_tasks: int, current_gain: Optional[float] = None) -> Tuple[np.ndarray, bool]:
        """
        動的な勾配重みを生成
        
        Parameters:
        -----------
        n_tasks : int
            タスク数
        current_gain : float, optional
            現在の情報利得（提案手法で使用）
            
        Returns:
        --------
        weights : array-like, shape=(n_tasks,)
            動的に生成された勾配重み
        switched : bool
            重みが切り替わったかどうか
        """
        if self.gradient_weights is not None:
            # 明示的に重みが指定されている場合はそれを使用
            return np.array(self.gradient_weights), False
        
        # 2タスクのみ対応
        if n_tasks != 2:
            return np.ones(n_tasks) / n_tasks, False
        
        base_weights = np.array([self.lambda_weight, 1 - self.lambda_weight])
        
        if self.weighting_strategy == "mtgbm":
            # 従来手法：50%の確率で反転
            if np.random.random() < 0.5:
                weights = base_weights[::-1]  # [1-λ, λ]
                switched = True
            else:
                weights = base_weights  # [λ, 1-λ]
                switched = False
        
        elif self.weighting_strategy == "proposed":
            # 提案手法：情報利得に基づく反転
            # 初期状態では基本重みを使用
            if not hasattr(self, '_current_weights'):
                self._current_weights = base_weights.copy()
            
            switched = False
            if current_gain is not None and current_gain < self.gain_threshold:
                # 利得が閾値を下回る場合は重みを反転
                self._current_weights = self._current_weights[::-1]
                switched = True
            
            weights = self._current_weights.copy()
        
        else:
            weights = base_weights
            switched = False
        
        # 正規化
        weights = weights / np.sum(weights)
        return weights, switched
    
    def _compute_split_gain_with_adaptive_weights(self, 
                                                 gradients: np.ndarray, 
                                                 hessians: np.ndarray,
                                                 feature_values: np.ndarray,
                                                 threshold: float,
                                                 depth: int,
                                                 feature_idx: int) -> Tuple[float, np.ndarray, bool]:
        """
        適応的重み付きの分割利得を計算
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            ヘシアン
        feature_values : array-like, shape=(n_samples,)
            分割対象の特徴量値
        threshold : float
            分割閾値
        depth : int
            現在の深さ
        feature_idx : int
            特徴量インデックス
            
        Returns:
        --------
        gain : float
            分割利得
        weights : array-like, shape=(n_tasks,)
            使用された重み
        switched : bool
            重みが切り替わったかどうか
        """
        left_mask = feature_values <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return -float('inf'), np.array([0.5, 0.5]), False
        
        # 重み戦略に応じた処理
        if self.weighting_strategy == "mtgbm":
            # 従来手法：ランダム重み選択
            base_weights = np.array([self.lambda_weight, 1 - self.lambda_weight])
            if np.random.random() < 0.5:
                weights = base_weights[::-1]
                switched = True
            else:
                weights = base_weights
                switched = False
        
        elif self.weighting_strategy == "proposed":
            # 提案手法：まず現在の重みで利得を計算
            temp_gain = self._compute_weighted_gain(
                gradients, hessians, left_mask, right_mask, self._current_weights
            )
            
            # 利得が閾値を下回る場合は重みを反転
            switched = False
            if temp_gain < self.gain_threshold:
                self._current_weights = self._current_weights[::-1]
                switched = True
            
            weights = self._current_weights.copy()
        
        else:
            weights = np.array([0.5, 0.5])
            switched = False
        
        # 最終的な利得計算
        gain = self._compute_weighted_gain(gradients, hessians, left_mask, right_mask, weights)
        
        # 履歴記録
        if self.track_gains:
            gain_info = {
                'iteration': self.current_iteration,
                'depth': depth,
                'feature': feature_idx,
                'threshold': threshold,
                'gain': gain,
                'weights': weights.copy(),
                'weight_switched': switched,
                'lambda_weight': self.lambda_weight,
                'gain_threshold': self.gain_threshold if self.weighting_strategy == "proposed" else None
            }
            self.split_gains.append(gain_info)
        
        return gain, weights, switched
    
    def _compute_weighted_gain(self, 
                              gradients: np.ndarray, 
                              hessians: np.ndarray,
                              left_mask: np.ndarray,
                              right_mask: np.ndarray,
                              weights: np.ndarray) -> float:
        """
        重み付き分割利得の計算
        """
        lambda_reg = 1.0  # L2正則化パラメータ
        total_gain = 0.0
        
        # 2タスクのみ対応
        for task_idx in range(min(2, gradients.shape[1])):
            # 重み付き統計量
            G = np.sum(gradients[:, task_idx]) * weights[task_idx]
            H = np.sum(hessians[:, task_idx]) * weights[task_idx]
            
            G_left = np.sum(gradients[left_mask, task_idx]) * weights[task_idx]
            H_left = np.sum(hessians[left_mask, task_idx]) * weights[task_idx]
            
            G_right = np.sum(gradients[right_mask, task_idx]) * weights[task_idx]
            H_right = np.sum(hessians[right_mask, task_idx]) * weights[task_idx]
            
            # XGBoost式利得計算
            if H_left > 0 and H_right > 0 and H > 0:
                gain_left = (G_left ** 2) / (H_left + lambda_reg)
                gain_right = (G_right ** 2) / (H_right + lambda_reg)
                gain_before = (G ** 2) / (H + lambda_reg)
                
                task_gain = 0.5 * (gain_left + gain_right - gain_before)
                total_gain += task_gain
        
        return total_gain
    
    def _get_mtgbm_weights(self) -> np.ndarray:
        """
        MTGBM戦略: λと1-λをランダム反転
        
        Returns:
        --------
        weights : array-like, shape=(2,)
            [λ, 1-λ] または [1-λ, λ] の重み
        """
        base_weights = np.array([self.lambda_weight, 1 - self.lambda_weight])
        if np.random.random() < 0.5:
            return base_weights[::-1]  # [1-λ, λ]
        return base_weights  # [λ, 1-λ]
    
    def _update_weights_by_strategy(self, current_gain):
        """戦略に応じた重み更新"""
        weight_switched = False
        
        if self.weighting_strategy == "mtgbm":
            # MTGBM: ランダム反転
            new_weights = self._get_mtgbm_weights()
            weight_switched = not np.array_equal(new_weights, self.current_weights)
            
        elif self.weighting_strategy == "proposed":
            # 提案手法: 利得閾値に基づく反転
            if current_gain < self.gain_threshold:
                new_weights = self.current_weights[::-1]  # 反転
                weight_switched = True
            else:
                new_weights = self.current_weights.copy()  # 維持
        else:
            new_weights = self.current_weights.copy()
        
        return new_weights, weight_switched
    
    def _compute_split_gain_with_weights(self, node_indices, gradients, hessians, 
                                       feature_idx, threshold, weights):
        """重み付きアンサンブル勾配での分割利得計算"""
        # アンサンブル勾配・ヘシアンの計算
        ensemble_gradients = np.zeros(len(node_indices))
        ensemble_hessians = np.zeros(len(node_indices))
        
        for task_idx in range(2):  # 2タスクのみ
            ensemble_gradients += weights[task_idx] * gradients[node_indices, task_idx]
            ensemble_hessians += weights[task_idx] * hessians[node_indices, task_idx]
        
        # 分割
        feature_values = self.X[node_indices, feature_idx]
        left_mask = feature_values <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return -float('inf')
        
        # XGBoost式利得計算
        G = np.sum(ensemble_gradients)
        H = np.sum(ensemble_hessians)
        G_L = np.sum(ensemble_gradients[left_mask])
        H_L = np.sum(ensemble_hessians[left_mask])
        G_R = np.sum(ensemble_gradients[right_mask])
        H_R = np.sum(ensemble_hessians[right_mask])
        
        if H_L <= 0 or H_R <= 0 or H <= 0:
            return -float('inf')
        
        gain = 0.5 * (
            (G_L ** 2) / (H_L + self.reg_lambda) +
            (G_R ** 2) / (H_R + self.reg_lambda) -
            (G ** 2) / (H + self.reg_lambda)
        )
        
        return gain
    
    def _record_split_attempt(self, depth, feature_idx, threshold, temp_gain, 
                            final_gain, old_weights, new_weights, switched):
        """分割試行の履歴記録"""
        if self.split_gains_history is not None:
            self.split_gains_history.append({
                'depth': depth,
                'feature': feature_idx,
                'threshold': threshold,
                'temp_gain': temp_gain,
                'final_gain': final_gain,
                'weights_before': old_weights,
                'weights_after': new_weights,
                'weight_switched': switched
            })
    
    def _find_best_split_with_adaptive_weights(self, node_indices, gradients, hessians, depth):
        """適応的重み付きの分割探索"""
        best_gain = -float('inf')
        best_split_info = None
        
        for feature_idx in range(self.n_features):
            # 分割候補の取得
            feature_values = self.X[node_indices, feature_idx]
            unique_values = np.unique(feature_values)
            
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Step 1: 現在の重みで仮の利得計算
                temp_gain = self._compute_split_gain_with_weights(
                    node_indices, gradients, hessians, feature_idx, threshold, 
                    self.current_weights
                )
                
                # Step 2: 戦略に応じた重み更新
                updated_weights, weight_switched = self._update_weights_by_strategy(temp_gain)
                
                # Step 3: 更新された重みで最終利得計算
                final_gain = self._compute_split_gain_with_weights(
                    node_indices, gradients, hessians, feature_idx, threshold,
                    updated_weights
                )
                
                # Step 4: 履歴記録
                if self.track_gains:
                    self._record_split_attempt(
                        depth, feature_idx, threshold, temp_gain, final_gain,
                        self.current_weights.copy(), updated_weights.copy(), weight_switched
                    )
                
                # Step 5: 最適分割更新
                if final_gain > best_gain:
                    best_gain = final_gain
                    best_split_info = {
                        'feature': feature_idx,
                        'threshold': threshold,
                        'gain': final_gain,
                        'weights_used': updated_weights.copy()
                    }
                    # 採用された重みで current_weights を更新
                    self.current_weights = updated_weights.copy()
        
        return best_split_info
    
    def _build_tree_with_adaptive_weights(self, node_indices, gradients, hessians, depth=0):
        """
        各分岐で適応的重み決定を行う再帰的木構築
        """
        # 停止条件チェック
        if (depth >= self.max_depth or 
            len(node_indices) < self.min_samples_split or
            len(node_indices) < 2 * self.min_samples_leaf):
            return self._create_leaf_node(node_indices, gradients, hessians)
        
        # 適応的重み付き分割探索
        best_split_info = self._find_best_split_with_adaptive_weights(
            node_indices, gradients, hessians, depth
        )
        
        if best_split_info is None or best_split_info['gain'] <= 0:
            return self._create_leaf_node(node_indices, gradients, hessians)
        
        # 分割実行
        feature_values = self.X[node_indices, best_split_info['feature']]
        left_mask = feature_values <= best_split_info['threshold']
        
        left_indices = node_indices[left_mask]
        right_indices = node_indices[~left_mask]
        
        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            return self._create_leaf_node(node_indices, gradients, hessians)
        
        # 再帰的に子ノード構築
        left_child = self._build_tree_with_adaptive_weights(left_indices, gradients, hessians, depth + 1)
        right_child = self._build_tree_with_adaptive_weights(right_indices, gradients, hessians, depth + 1)
        
        return DecisionTreeNode(
            feature_idx=best_split_info['feature'],
            threshold=best_split_info['threshold'],
            left=left_child,
            right=right_child
        )
    
    def _create_leaf_node(self, node_indices, gradients, hessians):
        """
        リーフノードを作成（各タスクの予測値を計算）
        """
        n_tasks = gradients.shape[1]
        leaf_values = np.zeros(n_tasks)
        
        for task_idx in range(n_tasks):
            G = np.sum(gradients[node_indices, task_idx])
            H = np.sum(hessians[node_indices, task_idx])
            
            if H > 0:
                leaf_values[task_idx] = -G / (H + self.reg_lambda)
            else:
                leaf_values[task_idx] = 0.0
        
        return DecisionTreeNode(values=leaf_values)
    
    def get_split_gain_analysis(self) -> dict:
        """
        分割利得の詳細分析を取得
        
        Returns:
        --------
        analysis : dict
            分割利得の詳細分析情報
        """
        if not self.track_split_gains:
            return {"error": "split gain tracking is disabled"}
        
        all_gains = []
        all_temp_gains = []
        all_switches = []
        iteration_stats = []
        
        for tree_info in self.trees:
            tree = tree_info['tree']
            if hasattr(tree, 'split_gains_history') and tree.split_gains_history:
                tree_gains = [g['final_gain'] for g in tree.split_gains_history]
                tree_temp_gains = [g['temp_gain'] for g in tree.split_gains_history]
                tree_switches = [g['weight_switched'] for g in tree.split_gains_history]
                
                all_gains.extend(tree_gains)
                all_temp_gains.extend(tree_temp_gains)
                all_switches.extend(tree_switches)
                
                # イテレーション統計
                iteration_stats.append({
                    'num_splits': len(tree_gains),
                    'gain_mean': np.mean(tree_gains) if tree_gains else 0,
                    'gain_std': np.std(tree_gains) if tree_gains else 0,
                    'weight_switches': np.sum(tree_switches),
                    'switch_rate': np.mean(tree_switches) if tree_switches else 0
                })
        
        if not all_gains:
            return {"error": "no split gain data available"}
        
        return {
            'gain_distribution': {
                'count': len(all_gains),
                'mean': np.mean(all_gains),
                'std': np.std(all_gains),
                'min': np.min(all_gains),
                'max': np.max(all_gains),
                'percentiles': {
                    '25%': np.percentile(all_gains, 25),
                    '50%': np.percentile(all_gains, 50),
                    '75%': np.percentile(all_gains, 75),
                    '90%': np.percentile(all_gains, 90),
                    '95%': np.percentile(all_gains, 95)
                }
            },
            'temp_gain_distribution': {
                'mean': np.mean(all_temp_gains),
                'std': np.std(all_temp_gains),
                'min': np.min(all_temp_gains),
                'max': np.max(all_temp_gains)
            },
            'weight_switch_summary': {
                'total_switches': np.sum(all_switches),
                'total_splits': len(all_switches),
                'overall_switch_rate': np.mean(all_switches),
                'avg_switches_per_tree': np.mean([s['weight_switches'] for s in iteration_stats])
            },
            'iteration_stats': iteration_stats
        }
