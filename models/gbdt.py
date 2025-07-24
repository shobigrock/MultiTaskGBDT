import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import time
import json
from datetime import datetime
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
    node_id : int
        ノードID
    depth : int
        ノードの深さ
    n_samples : int
        このノードのサンプル数
    cvr_rate : float
        このノードでのCVR率（タスク1の平均値）
    information_gain : float
        分割による情報利得（分割ノードの場合）
    """
    
    def __init__(self, node_id: int = 0, depth: int = 0):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.values = None
        
        # ログ用属性
        self.node_id = node_id
        self.depth = depth
        self.n_samples = 0
        self.cvr_rate = 0.0
        self.information_gain = 0.0
        
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
        
        # 子ノードから予測して次元数を取得
        n_tasks = None
        if np.any(left_mask):
            left_pred = self.left.predict(X[left_mask])
            n_tasks = left_pred.shape[1]
        elif np.any(right_mask):
            right_pred = self.right.predict(X[right_mask])  
            n_tasks = right_pred.shape[1]
        else:
            # fallback
            n_tasks = len(self.values) if self.values is not None else 2
        
        predictions = np.zeros((X.shape[0], n_tasks))
        
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
                 gain_threshold: float = 0.1,
                 track_split_gains: bool = True,
                 verbose_logging: bool = False,
                 is_dynamic_weight: bool = False,
                 gamma: float = 50.0,
                 delta: float = 0.5,
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
        gain_threshold : float, default=0.1
            情報利得閾値
        track_split_gains : bool, default=True
            分割利得の記録有無
        verbose_logging : bool, default=False
            ノードログの詳細出力有無
        is_dynamic_weight : bool, default=False
            アンサンブル重みの設定タイミング制御
        gamma : float, default=50.0
            勾配強調のハイパーパラメータ
        delta : float, default=0.5
            動的重み切り替えの情報利得閾値
        random_state : int, optional
            乱数シード
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.weighting_strategy = weighting_strategy
        self.gain_threshold = gain_threshold
        self.track_split_gains = track_split_gains
        self.verbose_logging = verbose_logging
        self.is_dynamic_weight = is_dynamic_weight
        self.gamma = gamma
        self.delta = delta
        self.root = None
        
        # 分割利得の記録
        self.split_gains = []
        self.current_iteration = 0
        
        # ノード情報のログ
        self.node_logs = []
        self.node_counter = 0
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, 
            X: np.ndarray, 
            gradients: np.ndarray, 
            hessians: np.ndarray, 
            task_correlations: Optional[np.ndarray] = None,
            y_true: Optional[np.ndarray] = None) -> 'MultiTaskDecisionTree':
        """
        マルチタスク決定木を学習
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配（素の値）
        hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのヘシアン（素の値）
        task_correlations : array-like, shape=(n_tasks, n_tasks), optional
            タスク間の相関行列
        y_true : array-like, shape=(n_samples, n_tasks), optional
            実際のラベル値（CVR率計算用）
            
        Returns:
        --------
        self : MultiTaskDecisionTree
            学習済みモデル
        """
        self.n_tasks = gradients.shape[1]
        
        # タスク相関が指定されていない場合は単位行列を使用
        if task_correlations is None:
            task_correlations = np.eye(self.n_tasks)
        
        # 1. 素の勾配・ヘシアンを正規化（予測値計算で使用）
        raw_gradients = self._compute_normalization_weights(gradients, target_mean=0.05, target_std=0.01)
        raw_hessians = self._compute_normalization_weights(hessians, target_mean=1.0, target_std=0.1)
        
        # 2. 重み戦略に応じたアンサンブル勾配・ヘシアンを作成（分割探索で使用）
        gamma_limited = min(self.gamma, 10.0)  # ガンマ値を制限して数値安定性を向上
        
        ensemble_gradients = raw_gradients.copy()
        
        # 乱数を用いて強調タスクを決定（CVR=0, CTR=1）
        if self.weighting_strategy == "mtgbm":
            emphasis_task = np.random.choice([0, 1])
            if emphasis_task == 0:  # CVR強調
                ensemble_gradients[:, 0] *= gamma_limited
            else:  # CTR強調
                ensemble_gradients[:, 1] *= gamma_limited

        # CVRタスク（インデックス0）の勾配にgammaを乗算        
        elif self.weighting_strategy == "proposed":
            ensemble_gradients[:, 0] *= gamma_limited

        # CTRタスク（インデックス1）の勾配にgammaを乗算
        elif self.weighting_strategy == "proposed_reverse":
            ensemble_gradients[:, 1] *= gamma_limited
        
        # CVRとCTRの重みを0.5:0.5に設定（均等重み）
        elif self.weighting_strategy == "half":
            ensemble_gradients[:, 0] *= 0.5  # CVR
            ensemble_gradients[:, 1] *= 0.5  # CTR
        
        # ルートノードを作成し、再帰的に木を構築
        # raw_gradients/hessians: 予測値計算用（素の値）
        # ensemble_gradients/hessians: 分割探索用（重み付け済み）
        self.root = DecisionTreeNode()
        self._build_tree(
            self.root, X, raw_gradients, raw_hessians, ensemble_gradients, raw_hessians,
            task_correlations, depth=0, y_true=y_true
        )
        
        return self
    
    def _compute_ensemble_weights_gradients_hessians(self, 
                                                    gradients: np.ndarray, 
                                                    hessians: np.ndarray,
                                                    is_cvr_emphasis: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        アンサンブル勾配・ヘシアン計算（アルゴリズム2）
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配（タスク0: CVR, タスク1: CTR）
        hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのヘシアン
        is_cvr_emphasis : bool, default=True
            CVRタスクを強調するかどうか（True: CVR強調, False: CTR強調）
            
        Returns:
        --------
        ensemble_gradients : array-like, shape=(n_samples,)
            アンサンブル勾配
        ensemble_hessians : array-like, shape=(n_samples,)
            アンサンブルヘシアン
        """
        n_samples, n_tasks = gradients.shape
        
        # 1. 勾配の正規化計算（平均0.05、標準偏差0.01）
        normalized_gradients = self._compute_normalization_weights(gradients, target_mean=0.05, target_std=0.01)
        
        # 2. ヘシアンの正規化計算（平均1.0、標準偏差0.1）
        normalized_hessians = self._compute_normalization_weights(hessians, target_mean=1.00, target_std=0.1)
        
        # 4. 勾配とヘシアンの加重和を計算
        ensemble_gradients = np.sum(normalized_gradients * self._current_weights, axis=1)
        ensemble_hessians = np.sum(normalized_hessians * self._current_weights, axis=1)
        
        return ensemble_gradients, ensemble_hessians
    
    def _compute_normalization_weights(self, data: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
        """
        データを指定された平均・標準偏差に線形変換により正規化
        
        Parameters:
        -----------
        data : array-like, shape=(n_samples, n_tasks)
            正規化対象データ
        target_mean : float
            目標平均値
        target_std : float
            目標標準偏差
            
        Returns:
        --------
        normalized_data : array-like, shape=(n_samples, n_tasks)
            正規化後のデータ
        """
        n_samples, n_tasks = data.shape
        normalized_data = np.zeros_like(data)
        
        for task_idx in range(n_tasks):
            task_data = data[:, task_idx]
            current_mean = np.mean(task_data)
            current_std = np.std(task_data)
            
            if current_std == 0:
                # 標準偏差が0の場合は目標平均値への単純シフト
                normalized_data[:, task_idx] = np.full_like(task_data, target_mean)
            else:
                # 線形変換: (data - current_mean) / current_std * target_std + target_mean
                normalized_data[:, task_idx] = (task_data - current_mean) / current_std * target_std + target_mean
        
        return normalized_data
    
    def _build_tree(self, 
                   node: DecisionTreeNode, 
                   X: np.ndarray, 
                   raw_gradients: np.ndarray, 
                   raw_hessians: np.ndarray,
                   ensemble_gradients: np.ndarray,
                   ensemble_hessians: np.ndarray,
                   task_correlations: np.ndarray,
                   depth: int,
                   y_true: Optional[np.ndarray] = None) -> None:
        """
        再帰的に決定木を構築（動的重み調整機能付き）
        
        Parameters:
        -----------
        node : DecisionTreeNode
            現在のノード
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        raw_gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの素の勾配（予測値計算用）
        raw_hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの素のヘシアン（予測値計算用）
        ensemble_gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのアンサンブル勾配（分割探索用）
        ensemble_hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのアンサンブルヘシアン（分割探索用）
        task_correlations : array-like, shape=(n_tasks, n_tasks)
            タスク間の相関行列
        depth : int
            現在の深さ
        y_true : array-like, shape=(n_samples, n_tasks), optional
            実際のラベル値（CVR計算用）
        """
        n_samples = X.shape[0]
        
        # ノード情報の設定
        node.node_id = self.node_counter
        node.depth = depth
        node.n_samples = n_samples
        self.node_counter += 1
        
        # CVR率の計算（タスク0をCVRタスクと仮定）
        if y_true is not None and self.n_tasks >= 2:
            node.cvr_rate = np.mean(y_true[:, 0]) if n_samples > 0 else 0.0
        else:
            # y_trueがない場合は素の勾配から推定（目安として）
            node.cvr_rate = max(0.0, min(1.0, -np.mean(raw_gradients[:, 0]) if self.n_tasks >= 2 else 0.0))
        
        # リーフノードの値を計算（素の勾配・ヘシアンを使用）
        node.values = self._compute_leaf_values(raw_gradients, raw_hessians, task_correlations, y_true)

        # 重み切り替えフラグ
        weight_switched = False

        # 停止条件をチェック
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_samples < 2 * self.min_samples_leaf):
            node.is_leaf = True
            # ログに記録
            self._log_node_info(node, is_leaf=True, weight_switched=weight_switched)
            return
        
        # アンサンブル勾配・ヘシアンを計算（分割探索用）
        current_ensemble_gradients = np.sum(ensemble_gradients, axis=1)
        current_ensemble_hessians = np.sum(ensemble_hessians, axis=1)
        

        # 初期のアンサンブル勾配・ヘシアンで最大情報利得を計算
        best_gain, best_feature_idx, best_threshold, best_left_indices, best_right_indices = self._search_best_split(
            X, current_ensemble_gradients, current_ensemble_hessians
        )

        # 有効な分割が見つからなかった場合
        if best_gain <= 0 or best_feature_idx is None:
            node.is_leaf = True
            # ログに記録
            self._log_node_info(node, is_leaf=True, weight_switched=weight_switched)
            # 葉ノード到達で終了
            return
        
        if self.weighting_strategy == "proposed":
            
            # 情報利得がdelta閾値を下回る場合、重みを切り替え
            if best_gain < self.delta:
                # CVRタスクの勾配を1/gamma倍、CTRタスクの勾配をgamma倍
                adjusted_ensemble_gradients = ensemble_gradients.copy()
                adjusted_ensemble_gradients[:, 0] /= self.gamma  # CVR
                adjusted_ensemble_gradients[:, 1] *= self.gamma  # CTR
                
                # 新しいアンサンブル勾配を計算（ヘシアンはそのまま）
                current_ensemble_gradients = np.sum(adjusted_ensemble_gradients, axis=1)
                weight_switched = True
        
                # 最適な分割を見つける
                best_gain, best_feature_idx, best_threshold, best_left_indices, best_right_indices = self._search_best_split(
                    X, current_ensemble_gradients, current_ensemble_hessians
                )

                # 有効な分割が見つからなかった場合
                if best_gain <= 0 or best_feature_idx is None:
                    node.is_leaf = True
                    # ログに記録
                    self._log_node_info(node, is_leaf=True, weight_switched=weight_switched)
                    # 葉ノード到達で終了
                    return
                
        elif self.weighting_strategy == "proposed_reverse":
            
            # 情報利得がdelta閾値を下回る場合、重みを切り替え
            if best_gain < self.delta:
                # CTRタスクの勾配を1/gamma倍、CVRタスクの勾配をgamma倍
                adjusted_ensemble_gradients = ensemble_gradients.copy()
                adjusted_ensemble_gradients[:, 1] /= self.gamma  # CTR
                adjusted_ensemble_gradients[:, 0] *= self.gamma  # CVR
                
                # 新しいアンサンブル勾配を計算（ヘシアンはそのまま）
                current_ensemble_gradients = np.sum(adjusted_ensemble_gradients, axis=1)
                weight_switched = True
        
                # 最適な分割を見つける
                best_gain, best_feature_idx, best_threshold, best_left_indices, best_right_indices = self._search_best_split(
                    X, current_ensemble_gradients, current_ensemble_hessians
                )

                # 有効な分割が見つからなかった場合
                if best_gain <= 0 or best_feature_idx is None:
                    node.is_leaf = True
                    # ログに記録
                    self._log_node_info(node, is_leaf=True, weight_switched=weight_switched)
                    # 葉ノード到達で終了
                    return

        # half戦略では動的重み調整を行わない（常に0.5:0.5を維持）
        elif self.weighting_strategy == "half":
            # 動的重み調整なし - 常に均等重み
            pass
              
        # ノードに分割情報を設定
        node.feature_idx = best_feature_idx
        node.threshold = best_threshold
        node.information_gain = best_gain
        
        # 分割ノードのログに記録
        self._log_node_info(node, is_leaf=False, weight_switched=weight_switched)
        
        # 左右の子ノードを作成
        node.left = DecisionTreeNode()
        node.right = DecisionTreeNode()
        
        # 左右の子ノードを再帰的に構築（素の勾配・ヘシアンとアンサンブル勾配・ヘシアンを両方渡す）
        self._build_tree(
            node.left, 
            X[best_left_indices], 
            raw_gradients[best_left_indices], 
            raw_hessians[best_left_indices],
            ensemble_gradients[best_left_indices],
            ensemble_hessians[best_left_indices],
            task_correlations,
            depth + 1,
            y_true[best_left_indices] if y_true is not None else None
        )
        
        self._build_tree(
            node.right, 
            X[best_right_indices], 
            raw_gradients[best_right_indices], 
            raw_hessians[best_right_indices],
            ensemble_gradients[best_right_indices],
            ensemble_hessians[best_right_indices],
            task_correlations,
            depth + 1,
            y_true[best_right_indices] if y_true is not None else None
        )
    
    def _compute_leaf_values(self, 
                            raw_gradients: np.ndarray, 
                            raw_hessians: np.ndarray,
                            task_correlations: np.ndarray,
                            y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        リーフノードの値を計算（素の勾配・ヘシアンを使用）
        
        Parameters:
        -----------
        raw_gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの素の勾配（重み付けされていない）
        raw_hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの素のヘシアン（重み付けされていない）
        task_correlations : array-like, shape=(n_tasks, n_tasks)
            タスク間の相関行列
        y_true : array-like, shape=(n_samples, n_tasks), optional
            実際のラベル値（利用可能な場合は実測値の平均を使用）
            
        Returns:
        --------
        leaf_values : array-like, shape=(n_tasks,)
            各タスクのリーフ値
        """
        n_samples = raw_gradients.shape[0]
        n_tasks = raw_gradients.shape[1]
        
        # 実測値が利用可能な場合は、それらの平均を使用
        if y_true is not None and n_samples > 0:
            # 葉ノード内の実測値の平均を計算
            leaf_values = np.mean(y_true, axis=0)
        else:
            # 実測値が利用できない場合は、勾配ブースティングの標準的な方法を使用
            # 各タスクの勾配とヘシアンの合計（素の値を使用）
            sum_gradients = np.sum(raw_gradients, axis=0)
            sum_hessians = np.sum(raw_hessians, axis=0)
            
            # L2正則化パラメータ（XGBoostのデフォルトに合わせて小さく設定）
            lambda_reg = 0.01
            
            # 各タスクのリーフ値を計算（標準的なXGBoostの式）
            leaf_values = np.zeros(n_tasks)
            for i in range(n_tasks):
                if sum_hessians[i] > 0:
                    # XGBoostの標準的な葉値計算: w* = -G / (H + λ)
                    leaf_values[i] = -sum_gradients[i] / (sum_hessians[i] + lambda_reg)
                else:
                    leaf_values[i] = 0.0
        
        return leaf_values
    

    def _search_best_split(self, 
                          X: np.ndarray, 
                          ensemble_gradients: np.ndarray, 
                          ensemble_hessians: np.ndarray) -> Tuple[float, int, float, np.ndarray, np.ndarray]:
        """
        アンサンブル勾配・ヘシアンを用いた最適分割探索
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        ensemble_gradients : array-like, shape=(n_samples,)
            アンサンブル勾配
        ensemble_hessians : array-like, shape=(n_samples,)
            アンサンブルヘシアン
            
        Returns:
        --------
        best_gain : float
            最良の情報利得
        best_feature_idx : int
            最良の特徴インデックス
        best_threshold : float
            最良の閾値
        best_left_indices : array-like
            左側のサンプルインデックス
        best_right_indices : array-like
            右側のサンプルインデックス
        """
        n_samples = X.shape[0]
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
                
                # 左右の子ノードの勾配とヘシアンの合計
                left_gradient = left_sum_gradients[i-1]
                left_hessian = left_sum_hessians[i-1]
                right_gradient = right_sum_gradients[i-1]
                right_hessian = right_sum_hessians[i-1]
                
                # 正則化項（XGBoostのデフォルトに合わせて小さく設定）
                lambda_reg = 0.01  # L2正則化パラメータ
                gamma_reg = 0.0   # 複雑性に対するペナルティ
                
                # 情報利得を計算（XGBoostの公式を使用）
                if left_hessian > 0 and right_hessian > 0 and (left_hessian + right_hessian) > 0:
                    gain = 0.5 * (
                        (left_gradient**2) / (left_hessian + lambda_reg) +
                        (right_gradient**2) / (right_hessian + lambda_reg) -
                        ((left_gradient + right_gradient)**2) / (left_hessian + right_hessian + lambda_reg)
                    )
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature_idx = feature_idx
                        best_threshold = (sorted_feature[i-1] + sorted_feature[i]) / 2
                        best_left_indices = sorted_indices[:i]
                        best_right_indices = sorted_indices[i:]
        
        return best_gain, best_feature_idx, best_threshold, best_left_indices, best_right_indices

    
    def _log_node_info(self, node: DecisionTreeNode, is_leaf: bool = False, weight_switched: bool = False):
        """
        ノード情報をログに記録
        
        Parameters:
        -----------
        node : DecisionTreeNode
            記録するノード
        is_leaf : bool, default=False
            リーフノードかどうか
        weight_switched : bool, default=False
            重みが切り替わったかどうか
        """
        log_entry = {
            'node_id': node.node_id,
            'depth': node.depth,
            'n_samples': node.n_samples,
            'cvr_rate': node.cvr_rate,
            'is_leaf': is_leaf,
            'feature_idx': node.feature_idx if not is_leaf else None,
            'threshold': node.threshold if not is_leaf else None,
            'information_gain': node.information_gain if not is_leaf else 0.0,
            'leaf_values': node.values.tolist() if hasattr(node.values, 'tolist') else node.values,
            'weight_switched': weight_switched,
        }
        
        self.node_logs.append(log_entry)
        
        # デバッグ用の詳細出力（オプション）
        if self.verbose_logging:
            if is_leaf:
                print(f"Leaf Node {node.node_id}: depth={node.depth}, samples={node.n_samples}, CVR={node.cvr_rate:.4f}")
            else:
                print(f"Split Node {node.node_id}: depth={node.depth}, samples={node.n_samples}, CVR={node.cvr_rate:.4f}, gain={node.information_gain:.4f}, feature={node.feature_idx}, threshold={node.threshold:.4f}, weight_switched={weight_switched}")
    
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
        # 通常の予測値をクリッピング（0-1の範囲）
        y_pred = self.predict(X)
        return np.clip(y_pred, 0, 1)
    
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
        gain_threshold: float = 0.1,
        track_split_gains: bool = True,
        is_dynamic_weight: bool = False,
        gamma: float = 50.0,
        delta: float = 0.5,
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
        gain_threshold : float, default=0.1
            情報利得の閾値δ（提案手法で使用）
        track_split_gains : bool, default=True
            分割利得の履歴を記録するかどうか
        is_dynamic_weight : bool, default=False
            アンサンブル重みの設定タイミング制御
            False: 各弱学習器のイテレーション前に決定（静的）
            True: 各分岐構造決定時に動的に決定
        gamma : float, default=50.0
            勾配強調のハイパーパラメータ（10-100推奨）
        delta : float, default=0.5
            動的重み切り替えの情報利得閾値
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
        self.gain_threshold = gain_threshold
        self.track_split_gains = track_split_gains
        self.is_dynamic_weight = is_dynamic_weight
        self.gamma = gamma
        self.delta = delta
        
        # 新しいパラメータの検証
        if self.weighting_strategy not in ["mtgbm", "proposed", "proposed_reverse", "half"]:
            raise ValueError(f"Unsupported weighting strategy: {self.weighting_strategy}. Use 'mtgbm', 'proposed', 'proposed_reverse', or 'half'.")
        
        # 履歴記録の初期化
        if self.track_split_gains:
            self.split_gains_history = []
            self.iteration_gains = []
            self.weight_switches = []
        
        # 木のリストを初期化
        self.trees = []
        
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
        # 実測値の平均を返す（MSEでもloglossでも同じ）
        # これにより木の予測値（実測値の平均）との整合性が保たれる
        return np.mean(y_multi, axis=0)
    
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
            # 予測値は既に確率値（0-1）なのでそのまま使用
            probs = np.clip(y_pred, 1e-7, 1 - 1e-7)  # 数値安定性のためクリッピング
            
            # 勾配: p - y (where p = predicted probability)
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
        
        # タスク数を取得
        n_tasks = y_multi.shape[1]
            
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
            
            # タスク間相関の計算
            task_correlations = self._compute_task_correlations(gradients)
            
            # サブサンプリング
            y_multi_subset = y_multi  # デフォルト値
            if self.subsample < 1.0:
                sample_indices = np.random.choice(
                    X.shape[0], 
                    size=int(X.shape[0] * self.subsample), 
                    replace=False
                )
                X_subset = X[sample_indices]
                gradients_subset = gradients[sample_indices]
                hessians_subset = hessians[sample_indices]
                y_multi_subset = y_multi[sample_indices]
            else:
                X_subset = X
                gradients_subset = gradients
                hessians_subset = hessians
                y_multi_subset = y_multi
            
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
                verbose_logging=False,  # デフォルトでは詳細ログを無効
                random_state=self.random_state,
                weighting_strategy=self.weighting_strategy,
                gamma=self.gamma,
                delta=self.delta,
            )
            
            # 木を学習（元の勾配・ヘシアンと加重和されたものを両方渡す）
            tree.fit(
                X_subset, 
                gradients_subset, 
                hessians_subset, 
                task_correlations=task_correlations,
                y_true=y_multi_subset
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
                
                if self.loss == "mse":
                    loss_value = np.mean((y_multi - y_pred) ** 2)
                    loss_name = "MSE"
                elif self.loss == "logloss":
                    # logloss計算：予測値は既に確率値なのでそのまま使用
                    y_pred_proba = np.clip(y_pred, 1e-7, 1 - 1e-7)
                    loss_value = -np.mean(y_multi * np.log(y_pred_proba) + (1 - y_multi) * np.log(1 - y_pred_proba))
                    loss_name = "LogLoss"
                else:
                    loss_value = np.mean((y_multi - y_pred) ** 2)
                    loss_name = "Loss"
                
                print(f"Iteration {i+1}/{self.n_estimators}, {loss_name}: {loss_value:.6f}, Time: {elapsed_time:.2f}s")
        
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
            各サンプルの予測値
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
        
        # 初期予測値と木の予測値の加算
        # 既に確率値（実測値の平均）なので、後処理は不要
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
        for tree_info in self.trees:
            tree = tree_info['tree']  # 辞書から取得
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
    
    def get_node_logs(self) -> List[Dict]:
        """
        全ての木のノードログを取得
        
        Returns:
        --------
        all_logs : List[Dict]
            全木の全ノードのログリスト
        """
        all_logs = []
        for i, tree_info in enumerate(self.trees):
            tree = tree_info['tree']
            for log in tree.node_logs:
                log_copy = log.copy()
                log_copy['tree_index'] = i
                all_logs.append(log_copy)
        return all_logs
    
    def print_node_summary(self, tree_index: Optional[int] = None):
        """
        ノードログの要約を出力
        
        Parameters:
        -----------
        tree_index : int, optional
            特定の木のインデックス（Noneの場合は全木）
        """
        if tree_index is not None:
            if tree_index >= len(self.trees):
                print(f"Tree index {tree_index} is out of range. Available trees: 0-{len(self.trees)-1}")
                return
            logs = self.trees[tree_index]['tree'].node_logs
            print(f"\n=== Tree {tree_index} Node Summary ===")
        else:
            logs = self.get_node_logs()
            print(f"\n=== All Trees Node Summary ({len(self.trees)} trees) ===")
        
        if not logs:
            print("No node logs available.")
            return
        
        # 統計情報の計算
        total_nodes = len(logs)
        leaf_nodes = sum(1 for log in logs if log['is_leaf'])
        split_nodes = total_nodes - leaf_nodes
        
        # CVR率の統計
        cvr_rates = [log['cvr_rate'] for log in logs]
        avg_cvr = np.mean(cvr_rates)
        min_cvr = np.min(cvr_rates)
        max_cvr = np.max(cvr_rates)
        
        # 情報利得の統計（分割ノードのみ）
        gains = [log['information_gain'] for log in logs if not log['is_leaf'] and log['information_gain'] > 0]
        if gains:
            avg_gain = np.mean(gains)
            max_gain = np.max(gains)
            min_gain = np.min(gains)
        else:
            avg_gain = max_gain = min_gain = 0.0
        
        print(f"Total nodes: {total_nodes}")
        print(f"Split nodes: {split_nodes}")
        print(f"Leaf nodes: {leaf_nodes}")
        print(f"CVR rate - Avg: {avg_cvr:.4f}, Min: {min_cvr:.4f}, Max: {max_cvr:.4f}")
        if gains:
            print(f"Information gain - Avg: {avg_gain:.4f}, Min: {min_gain:.4f}, Max: {max_gain:.4f}")
        
        # 深さ別の統計
        depth_stats = {}
        for log in logs:
            depth = log['depth']
            if depth not in depth_stats:
                depth_stats[depth] = {'count': 0, 'cvr_sum': 0, 'gain_sum': 0, 'gain_count': 0}
            depth_stats[depth]['count'] += 1
            depth_stats[depth]['cvr_sum'] += log['cvr_rate']
            if not log['is_leaf'] and log['information_gain'] > 0:
                depth_stats[depth]['gain_sum'] += log['information_gain']
                depth_stats[depth]['gain_count'] += 1
        
        print("\nDepth-wise statistics:")
        for depth in sorted(depth_stats.keys()):
            stats = depth_stats[depth]
            avg_cvr = stats['cvr_sum'] / stats['count']
            avg_gain = stats['gain_sum'] / stats['gain_count'] if stats['gain_count'] > 0 else 0.0
            print(f"  Depth {depth}: {stats['count']} nodes, CVR={avg_cvr:.4f}, Gain={avg_gain:.4f}")
    
    def save_logs_to_json(self, file_path: str):
        """
        ノードログをJSONファイルに保存
        
        Parameters:
        -----------
        file_path : str
            保存先のファイルパス
        """
        all_logs = self.get_node_logs()
        
        # タイムスタンプを追加
        for log in all_logs:
            log['timestamp'] = datetime.now().isoformat()
        
        # JSON形式で保存
        with open(file_path, 'w') as json_file:
            json.dump(all_logs, json_file, ensure_ascii=False, indent=4)
        
        print(f"Node logs saved to {file_path}")


class SingleTaskDecisionTree:
    """
    単一タスク決定木クラス（MTGBDTと同一アルゴリズム）
    
    Attributes:
    -----------
    max_depth : int
        木の最大深さ
    min_samples_split : int
        分割に必要な最小サンプル数
    min_samples_leaf : int
        リーフノードに必要な最小サンプル数
    root : DecisionTreeNode
        ルートノード
    """
    
    def __init__(self, 
                 max_depth: int = 3, 
                 min_samples_split: int = 2, 
                 min_samples_leaf: int = 1,
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
        random_state : int, optional
            乱数シード
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, 
            X: np.ndarray, 
            gradients: np.ndarray, 
            hessians: np.ndarray,
            y_true: Optional[np.ndarray] = None) -> 'SingleTaskDecisionTree':
        """
        単一タスク決定木を学習
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        gradients : array-like, shape=(n_samples,)
            各サンプルの勾配
        hessians : array-like, shape=(n_samples,)
            各サンプルのヘシアン
        y_true : array-like, shape=(n_samples,), optional
            実際のラベル値（リーフ値計算用）
            
        Returns:
        --------
        self : SingleTaskDecisionTree
            学習済みモデル
        """
        # ルートノードを作成し、再帰的に木を構築
        self.root = DecisionTreeNode()
        self._build_tree(self.root, X, gradients, hessians, depth=0, y_true=y_true)
        
        return self
    
    def _build_tree(self, 
                   node: DecisionTreeNode, 
                   X: np.ndarray, 
                   gradients: np.ndarray, 
                   hessians: np.ndarray,
                   depth: int,
                   y_true: Optional[np.ndarray] = None) -> None:
        """
        再帰的に決定木を構築
        
        Parameters:
        -----------
        node : DecisionTreeNode
            現在のノード
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        gradients : array-like, shape=(n_samples,)
            各サンプルの勾配
        hessians : array-like, shape=(n_samples,)
            各サンプルのヘシアン
        depth : int
            現在の深さ
        y_true : array-like, shape=(n_samples,), optional
            実際のラベル値
        """
        n_samples = X.shape[0]
        
        # ノード情報の設定
        node.depth = depth
        node.n_samples = n_samples
        
        # リーフノードの値を計算
        node.values = self._compute_leaf_values(gradients, hessians, y_true)
        
        # 停止条件をチェック
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_samples < 2 * self.min_samples_leaf):
            node.is_leaf = True
            return
        
        # 最適な分割を見つける
        best_gain, best_feature_idx, best_threshold, best_left_indices, best_right_indices = self._search_best_split(
            X, gradients, hessians
        )
        
        # 有効な分割が見つからなかった場合
        if best_gain <= 0 or best_feature_idx is None:
            node.is_leaf = True
            return
        
        # ノードに分割情報を設定
        node.feature_idx = best_feature_idx
        node.threshold = best_threshold
        node.information_gain = best_gain
        
        # 左右の子ノードを作成
        node.left = DecisionTreeNode()
        node.right = DecisionTreeNode()
        
        # 左右の子ノードを再帰的に構築
        self._build_tree(
            node.left, 
            X[best_left_indices], 
            gradients[best_left_indices], 
            hessians[best_left_indices],
            depth + 1,
            y_true[best_left_indices] if y_true is not None else None
        )
        
        self._build_tree(
            node.right, 
            X[best_right_indices], 
            gradients[best_right_indices], 
            hessians[best_right_indices],
            depth + 1,
            y_true[best_right_indices] if y_true is not None else None
        )
    
    def _compute_leaf_values(self, 
                            gradients: np.ndarray, 
                            hessians: np.ndarray,
                            y_true: Optional[np.ndarray] = None) -> float:
        """
        リーフノードの値を計算
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples,)
            各サンプルの勾配
        hessians : array-like, shape=(n_samples,)
            各サンプルのヘシアン
        y_true : array-like, shape=(n_samples,), optional
            実際のラベル値（利用可能な場合は実測値の平均を使用）
            
        Returns:
        --------
        leaf_value : float
            リーフ値
        """
        n_samples = len(gradients)
        
        # 実測値が利用可能な場合は、それらの平均を使用
        if y_true is not None and n_samples > 0:
            leaf_value = np.mean(y_true)
        else:
            # 実測値が利用できない場合は、勾配ブースティングの標準的な方法を使用
            sum_gradients = np.sum(gradients)
            sum_hessians = np.sum(hessians)
            
            # L2正則化パラメータ（XGBoostのデフォルトに合わせて小さく設定）
            lambda_reg = 0.01
            
            # XGBoostの標準的な葉値計算: w* = -G / (H + λ)
            if sum_hessians > 0:
                leaf_value = -sum_gradients / (sum_hessians + lambda_reg)
            else:
                leaf_value = 0.0
        
        return leaf_value
    
    def _search_best_split(self, 
                          X: np.ndarray, 
                          gradients: np.ndarray, 
                          hessians: np.ndarray) -> Tuple[float, int, float, np.ndarray, np.ndarray]:
        """
        勾配・ヘシアンを用いた最適分割探索
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        gradients : array-like, shape=(n_samples,)
            勾配
        hessians : array-like, shape=(n_samples,)
            ヘシアン
            
        Returns:
        --------
        best_gain : float
            最良の情報利得
        best_feature_idx : int
            最良の特徴インデックス
        best_threshold : float
            最良の閾値
        best_left_indices : array-like
            左側のサンプルインデックス
        best_right_indices : array-like
            右側のサンプルインデックス
        """
        n_samples = X.shape[0]
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
            sorted_gradients = gradients[sorted_indices]
            sorted_hessians = hessians[sorted_indices]
            
            # 累積和を計算
            left_sum_gradients = np.cumsum(sorted_gradients)
            left_sum_hessians = np.cumsum(sorted_hessians)
            
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
                
                # 左右の子ノードの勾配とヘシアンの合計
                left_gradient = left_sum_gradients[i-1]
                left_hessian = left_sum_hessians[i-1]
                right_gradient = right_sum_gradients[i-1]
                right_hessian = right_sum_hessians[i-1]
                
                # 正則化項（XGBoostのデフォルトに合わせて小さく設定）
                lambda_reg = 0.01  # L2正則化パラメータ
                
                # 情報利得を計算（XGBoostの公式を使用）
                if left_hessian > 0 and right_hessian > 0 and (left_hessian + right_hessian) > 0:
                    gain = 0.5 * (
                        (left_gradient**2) / (left_hessian + lambda_reg) +
                        (right_gradient**2) / (right_hessian + lambda_reg) -
                        ((left_gradient + right_gradient)**2) / (left_hessian + right_hessian + lambda_reg)
                    )
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature_idx = feature_idx
                        best_threshold = (sorted_feature[i-1] + sorted_feature[i]) / 2
                        best_left_indices = sorted_indices[:i]
                        best_right_indices = sorted_indices[i:]
        
        return best_gain, best_feature_idx, best_threshold, best_left_indices, best_right_indices
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        学習済みモデルで予測
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        predictions : array-like, shape=(n_samples,)
            各サンプルの予測値
        """
        if self.root is None:
            raise ValueError("Model has not been trained yet")
        
        return self._predict_single_node(self.root, X)
    
    def _predict_single_node(self, node: DecisionTreeNode, X: np.ndarray) -> np.ndarray:
        """
        単一ノードでの予測（単一タスク用）
        """
        if node.is_leaf:
            return np.full(X.shape[0], node.values)
        
        left_mask = X[:, node.feature_idx] <= node.threshold
        right_mask = ~left_mask
        
        predictions = np.zeros(X.shape[0])
        
        if np.any(left_mask):
            predictions[left_mask] = self._predict_single_node(node.left, X[left_mask])
        
        if np.any(right_mask):
            predictions[right_mask] = self._predict_single_node(node.right, X[right_mask])
            
        return predictions


class STGBDT(MTGBMBase):
    """
    単一タスク勾配ブースティング決定木（ST-GBDT）のスクラッチ実装
    MTGBDTと同一アルゴリズム、単一タスク版
    
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
    trees : list of SingleTaskDecisionTree
        学習済みの木のリスト
    initial_prediction : float
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
        loss: str = "mse",
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
        loss : str, default="mse"
            損失関数の種類（"mse": 平均二乗誤差, "logloss": クロスエントロピー損失）
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
        self.loss = loss
        
        # 木のリストを初期化
        self.trees = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_initial_prediction(self, y: np.ndarray) -> float:
        """
        初期予測値を計算
        
        Parameters:
        -----------
        y : array-like, shape=(n_samples,)
            ターゲット値
            
        Returns:
        --------
        initial_prediction : float
            初期予測値
        """
        # 実測値の平均を返す
        return np.mean(y)
    
    def _compute_gradients_hessians(self, y: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        勾配とヘシアンを計算
        
        Parameters:
        -----------
        y : array-like, shape=(n_samples,)
            ターゲット値
        y_pred : array-like, shape=(n_samples,)
            現在の予測値
            
        Returns:
        --------
        gradients : array-like, shape=(n_samples,)
            各サンプルの勾配
        hessians : array-like, shape=(n_samples,)
            各サンプルのヘシアン
        """
        if self.loss == "mse":
            # 二乗誤差損失の勾配とヘシアン
            gradients = 2 * (y_pred - y)
            hessians = np.ones_like(gradients) * 2
            
        elif self.loss == "logloss":
            # クロスエントロピー損失の勾配とヘシアン（二値分類）
            probs = np.clip(y_pred, 1e-7, 1 - 1e-7)  # 数値安定性のためクリッピング
            
            # 勾配: p - y
            gradients = probs - y
            
            # ヘシアン: p * (1 - p)
            hessians = probs * (1 - probs)
            # 数値安定性のため最小値を設定
            hessians = np.maximum(hessians, 1e-7)
            
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")
        
        return gradients, hessians
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'STGBDT':
        """
        単一タスクデータでモデルを学習
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y : array-like, shape=(n_samples,)
            ターゲット値
        **kwargs : dict
            追加のパラメータ
            
        Returns:
        --------
        self : STGBDT
            学習済みモデル
        """
        # 入力検証
        X = np.asarray(X)

        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array, got {y.ndim}D")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples")
            
        # 初期予測値を計算
        self.initial_prediction = self._compute_initial_prediction(y)
        
        # 現在の予測値を初期化
        y_pred = np.full_like(y, self.initial_prediction, dtype=float)
        
        # 学習開始時間
        start_time = time.time()
        
        # 各反復で木を構築
        for i in range(self.n_estimators):
            # 勾配とヘシアンを計算
            gradients, hessians = self._compute_gradients_hessians(y, y_pred)
            
            # サブサンプリング
            y_subset = y  # デフォルト値
            if self.subsample < 1.0:
                sample_indices = np.random.choice(
                    X.shape[0], 
                    size=int(X.shape[0] * self.subsample), 
                    replace=False
                )
                X_subset = X[sample_indices]
                gradients_subset = gradients[sample_indices]
                hessians_subset = hessians[sample_indices]
                y_subset = y[sample_indices]
            else:
                X_subset = X
                gradients_subset = gradients
                hessians_subset = hessians
                y_subset = y
            
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
            
            # 新しい木を構築
            tree = SingleTaskDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            
            # 木を学習
            tree.fit(X_subset, gradients_subset, hessians_subset, y_true=y_subset)
            
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
                
                if self.loss == "mse":
                    loss_value = np.mean((y - y_pred) ** 2)
                    loss_name = "MSE"
                elif self.loss == "logloss":
                    # logloss計算
                    y_pred_proba = np.clip(y_pred, 1e-7, 1 - 1e-7)
                    loss_value = -np.mean(y * np.log(y_pred_proba) + (1 - y) * np.log(1 - y_pred_proba))
                    loss_name = "LogLoss"
                else:
                    loss_value = np.mean((y - y_pred) ** 2)
                    loss_name = "Loss"
                
                print(f"Iteration {i+1}/{self.n_estimators}, {loss_name}: {loss_value:.6f}, Time: {elapsed_time:.2f}s")
        
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
        y_pred : array-like, shape=(n_samples,)
            各サンプルの予測値
        """
        # 入力検証
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        
        if not self.trees:
            raise ValueError("Model has not been trained yet")
        
        # 初期予測値
        y_pred = np.full(X.shape[0], self.initial_prediction, dtype=float)
        
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
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        確率予測を行う（logloss用）
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        y_pred_proba : array-like, shape=(n_samples, n_tasks, 2)
            各サンプルの確率予測値（クラス0とクラス1の確率）
        """
        # 通常の予測値をクリッピング（0-1の範囲）
        y_pred = self.predict(X)
        y_pred_proba = np.clip(y_pred, 0, 1)
        
        # [1-p, p] の形式で返却（sklearn互換）
        return np.stack([1 - y_pred_proba, y_pred_proba], axis=-1)


class MTRF:
    """
    マルチタスクランダムフォレスト（MT-RF）の実装
    
    MultiTaskDecisionTreeを弱学習器とするランダムフォレスト。
    MTGBDTのランダムフォレスト版として、バギング手法を採用。
    
    Attributes:
    -----------
    n_estimators : int
        木の数
    max_depth : int
        各木の最大深さ
    min_samples_split : int
        分割に必要な最小サンプル数
    min_samples_leaf : int
        リーフノードに必要な最小サンプル数
    max_features : str, int, float or None
        各木で使用する特徴数の指定
    bootstrap : bool
        ブートストラップサンプリングを行うかどうか
    weighting_strategy : str
        重み戦略（"mtgbm", "proposed", "half"など）
    gamma : float
        勾配強調のハイパーパラメータ（proposed戦略用）
    delta : float
        動的重み切り替えの情報利得閾値（proposed戦略用）
    random_state : int or None
        乱数シード
    trees : list
        学習済みの木のリスト
    initial_predictions : array-like
        初期予測値（各タスクの平均値）
    loss : str
        損失関数の種類
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 max_features: Union[str, int, float, None] = None,
                 bootstrap: bool = True,
                 weighting_strategy: str = "mtgbm",
                 gamma: float = 50.0,
                 delta: float = 0.5,
                 loss: str = "logloss",
                 random_state: Optional[int] = None):
        """
        初期化メソッド
        
        Parameters:
        -----------
        n_estimators : int, default=100
            木の数
        max_depth : int, default=3
            各木の最大深さ
        min_samples_split : int, default=2
            分割に必要な最小サンプル数
        min_samples_leaf : int, default=1
            リーフノードに必要な最小サンプル数
        max_features : str, int, float or None, default=None
            各木で使用する特徴数の指定
            - None: 全特徴を使用
            - "sqrt": sqrt(n_features)
            - "log2": log2(n_features)
            - int: 指定した数の特徴
            - float: 指定した割合の特徴
        bootstrap : bool, default=True
            ブートストラップサンプリングを行うかどうか
        weighting_strategy : str, default="mtgbm"
            重み戦略（"mtgbm", "proposed", "half"など）
        gamma : float, default=50.0
            勾配強調のハイパーパラメータ（proposed戦略用）
        delta : float, default=0.5
            動的重み切り替えの情報利得閾値（proposed戦略用）
        loss : str, default="logloss"
            損失関数の種類（"mse", "logloss"）
        random_state : int or None, default=None
            乱数シード
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.weighting_strategy = weighting_strategy
        self.gamma = gamma
        self.delta = delta
        self.loss = loss
        self.random_state = random_state
        
        # 学習済みモデルの保存
        self.trees = []
        self.initial_predictions = None
        
        # 乱数生成器の設定
        if random_state is not None:
            np.random.seed(random_state)
            
    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        入力データの検証
        
        Parameters:
        -----------
        X : array-like
            入力特徴量
        y : array-like or None
            ターゲット値
            
        Returns:
        --------
        X : array-like
            検証済み入力特徴量
        y : array-like or None
            検証済みターゲット値
        """
        X = np.asarray(X)
        if y is not None:
            y = np.asarray(y)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
        return X, y
    
    def _get_n_features_for_tree(self, n_total_features: int) -> int:
        """
        各木で使用する特徴数を計算
        
        Parameters:
        -----------
        n_total_features : int
            全特徴数
            
        Returns:
        --------
        n_features : int
            使用する特徴数
        """
        if self.max_features is None:
            return n_total_features
        elif self.max_features == "sqrt":
            return int(np.sqrt(n_total_features))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_total_features)))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_total_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_total_features))
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")
    
    def _compute_initial_predictions(self, y_multi: np.ndarray) -> np.ndarray:
        """
        初期予測値を計算（各タスクの平均値）
        
        Parameters:
        -----------
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクのターゲット値
            
        Returns:
        --------
        initial_predictions : array-like, shape=(n_tasks,)
            各タスクの初期予測値
        """
        return np.mean(y_multi, axis=0)
    
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
            勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            ヘシアン
        """
        if self.loss == "mse":
            # 平均二乗誤差の場合
            gradients = y_pred - y_multi  # MSEの勾配
            hessians = np.ones_like(y_multi)  # MSEのヘシアン
        
        elif self.loss == "logloss":
            # クロスエントロピー損失の場合
            y_pred_safe = np.clip(y_pred, 1e-7, 1 - 1e-7)  # 数値安定性のため
            gradients = y_pred_safe - y_multi  # loglossの勾配
            hessians = y_pred_safe * (1 - y_pred_safe)  # loglossのヘシアン
        
        else:
            raise ValueError(f"Unknown loss function: {self.loss}")
            
        return gradients, hessians
    
    def fit(self, X: np.ndarray, y_multi: np.ndarray, **kwargs) -> 'MTRF':
        """
        マルチタスクランダムフォレストを学習
        
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
        self : MTRF
            学習済みモデル
        """
        # 入力検証
        X, y_multi = self._validate_input(X, y_multi)
        
        # 初期予測値を計算（各タスクの平均値）
        self.initial_predictions = self._compute_initial_predictions(y_multi)
        
        # 初期予測値から勾配・ヘシアンを計算
        y_pred_initial = np.tile(self.initial_predictions, (X.shape[0], 1))
        gradients_initial, hessians_initial = self._compute_gradients_hessians(y_multi, y_pred_initial)
        
        # タスク間相関の計算（ランダムフォレストでは各木が独立なため、無相関を仮定）
        n_tasks = y_multi.shape[1]
        task_correlations = np.eye(n_tasks)
        
        # 学習開始時間
        start_time = time.time()
        
        # 各木を学習
        self.trees = []
        for i in range(self.n_estimators):
            # 各木用のサンプリング（ブートストラップ + 特徴選択）
            n_samples = X.shape[0]
            
            # ブートストラップサンプリング
            if self.bootstrap:
                sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            else:
                sample_indices = np.arange(n_samples)
            
            # 特徴サンプリング
            n_features_for_tree = self._get_n_features_for_tree(X.shape[1])
            feature_indices = np.random.choice(X.shape[1], size=n_features_for_tree, replace=False)
            
            # サンプリング実行
            X_subset = X[np.ix_(sample_indices, feature_indices)]
            y_subset = y_multi[sample_indices]
            
            # この木の重み戦略を決定
            if self.weighting_strategy == "mtgbm":
                tree_weight_strategy = np.random.choice(["task1_priority", "task2_priority"])
            else:
                tree_weight_strategy = self.weighting_strategy
            
            # サブセットでの勾配・ヘシアンを計算
            y_pred_subset = np.tile(self.initial_predictions, (X_subset.shape[0], 1))
            gradients_subset, hessians_subset = self._compute_gradients_hessians(y_subset, y_pred_subset)
            
            # 木を構築
            tree = MultiTaskDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                weighting_strategy=tree_weight_strategy,
                gamma=self.gamma,
                delta=self.delta,
                random_state=None  # 各木で異なる乱数を使用
            )
            
            # 木を学習
            tree.fit(
                X_subset,
                gradients_subset,
                hessians_subset,
                task_correlations=task_correlations,
                y_true=y_subset
            )
            
            # 木と特徴インデックスを保存
            self.trees.append({
                'tree': tree,
                'feature_indices': feature_indices
            })
            
            # 進捗表示
            if (i + 1) % 20 == 0 or i == 0 or i == self.n_estimators - 1:
                elapsed_time = time.time() - start_time
                print(f"Tree {i+1}/{self.n_estimators} trained, Time: {elapsed_time:.2f}s")
        
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
            各サンプルの予測値（各木の単純平均）
        """
        # 入力検証
        X, _ = self._validate_input(X)
        
        if not self.trees:
            raise ValueError("Model has not been trained yet")
        
        # 各木の予測を収集
        tree_predictions = []
        
        for tree_info in self.trees:
            tree = tree_info['tree']
            feature_indices = tree_info['feature_indices']
            
            # 特徴サブセットで予測
            X_subset = X[:, feature_indices]
            pred = tree.predict(X_subset)
            tree_predictions.append(pred)
        
        # 各木の予測の単純平均
        y_pred = np.mean(tree_predictions, axis=0)
        
        # 初期予測値を加算（ランダムフォレストでは通常不要だが、整合性のため）
        # ただし、木の予測が既に初期予測からの差分ではなく絶対値なので、単純平均のみ
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        確率予測を行う（logloss用）
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        y_pred_proba : array-like, shape=(n_samples, n_tasks, 2)
            各サンプルの確率予測値（クラス0とクラス1の確率）
        """
        # 通常の予測値をクリッピング（0-1の範囲）
        y_pred = self.predict(X)
        y_pred_proba = np.clip(y_pred, 0, 1)
        
        # [1-p, p] の形式で返却（sklearn互換）
        return np.stack([1 - y_pred_proba, y_pred_proba], axis=-1)