"""
マルチタスク勾配ブースティング決定木（MT-GBDT）の実装

このモジュールは、論文「MT-GBM: A multi-task Gradient Boosting Machine with shared decision trees」
に基づいたマルチタスク勾配ブースティング決定木の実装を提供します。

論文の主要アルゴリズム：
- アルゴリズム1: MT-GBM全体の学習アルゴリズム
- アルゴリズム2: 共有決定木の構築
- アルゴリズム3: タスク間相関を考慮したリーフ値の計算

著者: Ying et al. (2022)
リファクタリング: 2025年6月
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import time


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
    split_gain : float
        このノードでの分割利得
    n_samples : int
        このノードに含まれるサンプル数
    n_tasks_inferred : int
        このノードが扱うタスク数（予測時に使用）
    """
    
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = False
        self.values = None
        self.split_gain = 0.0
        self.n_samples = 0
        self.n_tasks_inferred = None
        
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
            # リーフノードの場合、全サンプルに同じ値を返す
            return np.tile(self.values, (X.shape[0], 1))
        
        # タスク数を取得
        n_tasks = self._get_n_tasks()
        
        # 予測値を初期化
        predictions = np.zeros((X.shape[0], n_tasks))
        
        # 分割条件に基づいて左右の子ノードに振り分け
        left_mask = X[:, self.feature_idx] <= self.threshold
        right_mask = ~left_mask
        
        if np.any(left_mask):
            predictions[left_mask] = self.left.predict(X[left_mask])
        
        if np.any(right_mask):
            predictions[right_mask] = self.right.predict(X[right_mask])
            
        return predictions
    
    def _get_n_tasks(self) -> int:
        """
        このノードが扱うタスク数を取得
        
        Returns:
        --------
        n_tasks : int
            タスク数
        """
        if self.values is not None:
            return self.values.shape[0]
        elif self.n_tasks_inferred is not None:
            return self.n_tasks_inferred
        elif self.left is not None and self.left.values is not None:
            return self.left.values.shape[0]
        elif self.right is not None and self.right.values is not None:
            return self.right.values.shape[0]
        else:
            # フォールバック（通常は起こらないはず）
            return 1


class MultiTaskDecisionTree:
    """
    マルチタスク決定木クラス
    
    論文のアルゴリズム2と3に基づいて、複数タスクに対して共有された決定木を構築します。
    分割点の探索にはタスク間の相関を考慮し、リーフ値の計算には同時最適化を行います。
    
    Attributes:
    -----------
    max_depth : int
        木の最大深さ
    min_samples_split : int
        分割に必要な最小サンプル数
    min_samples_leaf : int
        リーフノードに必要な最小サンプル数
    gamma : float
        メインタスク強調パラメータ
    lambda_l2 : float
        L2正則化パラメータ (葉の重み用)
    lambda_correlation : float
        タスク間相関のペナルティ/報酬係数 (葉の重み用)
    min_split_gain : float
        分割に必要な最小利得
    n_tasks : int
        タスク数
    root : DecisionTreeNode
        ルートノード
    """
    
    def __init__(self, 
                 max_depth: int = 3, 
                 min_samples_split: int = 2, 
                 min_samples_leaf: int = 1,
                 gamma: float = 1.0,
                 lambda_l2: float = 1.0,
                 lambda_correlation: float = 0.1,
                 min_split_gain: float = 0.0,
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
        gamma : float, default=1.0
            メインタスク強調パラメータ（分割利得計算時の重み）
        lambda_l2 : float, default=1.0
            L2正則化パラメータ (葉の重み用)
        lambda_correlation : float, default=0.1
            タスク間相関のペナルティ/報酬係数 (葉の重み用)
        min_split_gain : float, default=0.0
            分割に必要な最小利得
        random_state : int, optional
            乱数シード
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.gamma = gamma
        self.lambda_l2 = lambda_l2
        self.lambda_correlation = lambda_correlation
        self.min_split_gain = min_split_gain
        self.root = None
        self.n_tasks = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, 
            X: np.ndarray, 
            gradients: np.ndarray, 
            hessians: np.ndarray, 
            task_correlations: Optional[np.ndarray] = None,
            main_task_idx: Optional[int] = None) -> 'MultiTaskDecisionTree':
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
        task_correlations : array-like, shape=(n_tasks, n_tasks), optional
            タスク間の相関行列
        main_task_idx : int, optional
            メインタスクのインデックス（指定がない場合はランダムに選択）
            
        Returns:
        --------
        self : MultiTaskDecisionTree
            学習済みモデル
        """
        # 入力検証
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if gradients.ndim != 2:
            raise ValueError("gradients must be 2-dimensional")
        if hessians.ndim != 2:
            raise ValueError("hessians must be 2-dimensional")
        if X.shape[0] != gradients.shape[0] or X.shape[0] != hessians.shape[0]:
            raise ValueError("X, gradients, and hessians must have the same number of samples")
        if gradients.shape[1] != hessians.shape[1]:
            raise ValueError("gradients and hessians must have the same number of tasks")
        
        self.n_tasks = gradients.shape[1]
        
        # タスク相関が指定されていない場合は単位行列を使用
        if task_correlations is None:
            task_correlations = np.eye(self.n_tasks)
        elif task_correlations.shape != (self.n_tasks, self.n_tasks):
            raise ValueError("task_correlations shape must match the number of tasks")
        
        # メインタスクが指定されていない場合はランダムに選択
        if main_task_idx is None:
            main_task_idx = np.random.randint(0, self.n_tasks)
        elif main_task_idx < 0 or main_task_idx >= self.n_tasks:
            raise ValueError(f"main_task_idx must be between 0 and {self.n_tasks-1}")
        
        # ルートノードを作成し、再帰的に木を構築
        self.root = DecisionTreeNode()
        self.root.n_tasks_inferred = self.n_tasks
        self._build_tree(
            self.root, X, gradients, hessians, 
            task_correlations, main_task_idx, depth=0
        )
        
        return self
    
    def _build_tree(self, 
                   node: DecisionTreeNode, 
                   X: np.ndarray, 
                   gradients: np.ndarray, 
                   hessians: np.ndarray,
                   task_correlations: np.ndarray,
                   main_task_idx: int,
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
        task_correlations : array-like, shape=(n_tasks, n_tasks)
            タスク間の相関行列
        main_task_idx : int
            メインタスクのインデックス
        depth : int
            現在の深さ
        """
        n_samples, n_features = X.shape
        node.n_samples = n_samples
        node.n_tasks_inferred = self.n_tasks
        
        # リーフノードの値を計算 (アルゴリズム3)
        node.values = self._compute_leaf_values(gradients, hessians, task_correlations)
        
        # 停止条件をチェック
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_samples < 2 * self.min_samples_leaf):
            node.is_leaf = True
            return
        
        # 最適な分割を見つける（アルゴリズム2）
        best_split = self._find_best_split(X, gradients, hessians, task_correlations, main_task_idx)
        
        # 有効な分割が見つからなかった場合
        if best_split["gain"] <= self.min_split_gain:
            node.is_leaf = True
            return
        
        # ノードに分割情報を設定
        node.feature_idx = best_split["feature_idx"]
        node.threshold = best_split["threshold"]
        node.split_gain = best_split["gain"]
        
        # 左右の子ノードのサンプルを分割
        left_mask = X[:, node.feature_idx] <= node.threshold
        right_mask = ~left_mask
        
        # 左右の子ノードを作成
        node.left = DecisionTreeNode()
        node.left.n_tasks_inferred = self.n_tasks
        node.right = DecisionTreeNode()
        node.right.n_tasks_inferred = self.n_tasks
        
        # 左右の子ノードを再帰的に構築
        self._build_tree(
            node.left, 
            X[left_mask], 
            gradients[left_mask], 
            hessians[left_mask],
            task_correlations,
            main_task_idx,
            depth + 1
        )
        
        self._build_tree(
            node.right, 
            X[right_mask], 
            gradients[right_mask], 
            hessians[right_mask],
            task_correlations,
            main_task_idx,
            depth + 1
        )
    
    def _find_best_split(self, 
                        X: np.ndarray, 
                        gradients: np.ndarray, 
                        hessians: np.ndarray,
                        task_correlations: np.ndarray,
                        main_task_idx: int) -> Dict:
        """
        最適な分割点を探索 (アルゴリズム2)
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのヘシアン
        task_correlations : array-like, shape=(n_tasks, n_tasks)
            タスク間の相関行列
        main_task_idx : int
            メインタスクのインデックス
            
        Returns:
        --------
        best_split : dict
            最適な分割情報
        """
        n_samples, n_features = X.shape
        
        # 最適な分割情報を初期化
        best_split = {
            "gain": -float("inf"),
            "feature_idx": None,
            "threshold": None
        }
        
        # タスク重み (メインタスク強調)
        task_weights = np.ones(self.n_tasks)
        task_weights[main_task_idx] *= self.gamma
        
        # 親ノードの目的関数値 (分割前)
        G_total = np.sum(gradients, axis=0)
        H_total = np.sum(hessians, axis=0)
        obj_parent = self._calculate_objective(G_total, H_total, task_correlations, task_weights)
        
        # 各特徴について最適な分割を探索
        for feature_idx in range(n_features):
            # ユニークな特徴量値でソートし、分割候補とする
            unique_values = np.unique(X[:, feature_idx])
            if len(unique_values) <= 1:
                continue
            
            # 分割閾値の候補
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
            
            for threshold_val in thresholds:
                # 左右の子ノードのサンプルを分割
                left_mask = X[:, feature_idx] <= threshold_val
                right_mask = ~left_mask
                
                # 最小サンプル数のチェック
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # 左右の子ノードの勾配・ヘシアン合計
                G_left = np.sum(gradients[left_mask], axis=0)
                H_left = np.sum(hessians[left_mask], axis=0)
                G_right = np.sum(gradients[right_mask], axis=0)
                H_right = np.sum(hessians[right_mask], axis=0)
                
                # 左右の子ノードの目的関数値
                obj_left = self._calculate_objective(G_left, H_left, task_correlations, task_weights)
                obj_right = self._calculate_objective(G_right, H_right, task_correlations, task_weights)
                
                # 分割利得 = 親ノードの目的関数値 - (左の子ノードの目的関数値 + 右の子ノードの目的関数値)
                gain = obj_parent - (obj_left + obj_right)
                
                # より良い分割が見つかった場合は更新
                if gain > best_split["gain"]:
                    best_split["gain"] = gain
                    best_split["feature_idx"] = feature_idx
                    best_split["threshold"] = threshold_val
        
        return best_split
    
    def _calculate_objective(self, 
                            G: np.ndarray, 
                            H: np.ndarray, 
                            task_correlations: np.ndarray, 
                            task_weights: np.ndarray) -> float:
        """
        指定された勾配・ヘシアン合計に対する目的関数値を計算
        
        Parameters:
        -----------
        G : array-like, shape=(n_tasks,)
            各タスクの勾配合計
        H : array-like, shape=(n_tasks,)
            各タスクのヘシアン合計
        task_correlations : array-like, shape=(n_tasks, n_tasks)
            タスク間の相関行列
        task_weights : array-like, shape=(n_tasks,)
            各タスクの重み
            
        Returns:
        --------
        obj_val : float
            目的関数値
        """
        if G.shape[0] == 0:
            return 0.0
        
        # 行列Aとベクトルbを構築
        A = np.diag(H + self.lambda_l2)
        for k in range(self.n_tasks):
            for l in range(self.n_tasks):
                if k != l:
                    A[k, l] += self.lambda_correlation * task_correlations[k, l]
        
        b = -G
        
        try:
            # 連立一次方程式 Aw = b を解く
            optimal_weights = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # 特異行列の場合は対角成分のみで近似
            optimal_weights = -G / (H + self.lambda_l2 + 1e-9)
            optimal_weights = np.nan_to_num(optimal_weights)
        
        # 目的関数値: L = G^T w + 0.5 * w^T A w
        # 最適なwを代入すると L = -0.5 * G^T A^-1 G = 0.5 * G^T w
        obj_val = 0.5 * np.dot(G, optimal_weights)
        
        return obj_val
    
    def _compute_leaf_values(self, 
                            gradients: np.ndarray, 
                            hessians: np.ndarray,
                            task_correlations: np.ndarray) -> np.ndarray:
        """
        リーフノードの値を計算（アルゴリズム3、論文 式(6)(7)）
        
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
        n_samples, n_tasks = gradients.shape
        
        if n_samples == 0:
            return np.zeros(n_tasks)
        
        # 各タスクの勾配とヘシアンの合計
        sum_gradients = np.sum(gradients, axis=0)
        sum_hessians = np.sum(hessians, axis=0)
        
        # 行列Aとベクトルbを構築
        # A_kl = δ_kl * (H_k + λ_2) + (1 - δ_kl) * λ_c * ρ_kl
        # b_k = -G_k
        A = np.zeros((n_tasks, n_tasks))
        for k in range(n_tasks):
            A[k, k] = sum_hessians[k] + self.lambda_l2
            for l in range(n_tasks):
                if k != l:
                    A[k, l] = self.lambda_correlation * task_correlations[k, l]
        
        b = -sum_gradients
        
        try:
            # 連立一次方程式 Aw = b を解く
            leaf_values = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # 特異行列の場合は対角成分のみで近似
            leaf_values = -sum_gradients / (sum_hessians + self.lambda_l2 + 1e-9)
            leaf_values = np.nan_to_num(leaf_values)
        
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
        
        # 入力検証
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return self.root.predict(X)


class MTGBM:
    """
    マルチタスク勾配ブースティングマシン（MT-GBM）
    
    論文「MT-GBM: A multi-task Gradient Boosting Machine with shared decision trees」に基づいた
    マルチタスク学習のための勾配ブースティングマシンの実装です。
    
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
    gamma : float
        メインタスク強調パラメータ
    lambda_l2 : float
        L2正則化パラメータ
    lambda_correlation : float
        タスク間相関のペナルティ/報酬係数
    min_split_gain : float
        分割に必要な最小利得
    subsample : float
        各木を構築する際のサンプリング率
    colsample_bytree : float
        各木を構築する際の特徴サンプリング率
    main_task_strategy : str
        メインタスク選択戦略
    normalize_gradients : bool
        勾配・ヘシアンを標準化するかどうか
    loss : str or callable
        損失関数
    random_state : int or None
        乱数シード
    verbose : int
        詳細出力レベル
    trees : list
        学習済みの木のリスト
    initial_predictions : array-like
        初期予測値
    n_tasks : int
        タスク数
    feature_importances_ : array-like
        特徴量重要度
    """
    
    def __init__(self, 
                 n_estimators: int = 100, 
                 learning_rate: float = 0.1, 
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 gamma: float = 1.0,
                 lambda_l2: float = 1.0,
                 lambda_correlation: float = 0.1,
                 min_split_gain: float = 0.0,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 main_task_strategy: str = 'random',
                 normalize_gradients: bool = False,
                 loss: Union[str, Callable] = "mse",
                 random_state: Optional[int] = None,
                 verbose: int = 0):
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
        gamma : float, default=1.0
            メインタスク強調パラメータ（分割利得計算時の重み）
        lambda_l2 : float, default=1.0
            L2正則化パラメータ (葉の重み用)
        lambda_correlation : float, default=0.1
            タスク間相関のペナルティ/報酬係数 (葉の重み用)
        min_split_gain : float, default=0.0
            分割に必要な最小利得
        subsample : float, default=1.0
            各木を構築する際のサンプリング率
        colsample_bytree : float, default=1.0
            各木を構築する際の特徴サンプリング率
        main_task_strategy : str, default='random'
            メインタスク選択戦略（'random': ランダム選択, 'rotate': 順番に選択, 'fixed': 常に最初のタスク）
        normalize_gradients : bool, default=False
            勾配・ヘシアンを標準化するかどうか
        loss : str or callable, default="mse"
            損失関数の種類（"mse": 平均二乗誤差, "logloss": クロスエントロピー損失）
            または、カスタム損失関数（勾配とヘシアンを返す関数）
        random_state : int, optional
            乱数シード
        verbose : int, default=0
            詳細出力レベル（0: 出力なし, 1: 進捗表示, 2: 詳細表示）
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.gamma = gamma
        self.lambda_l2 = lambda_l2
        self.lambda_correlation = lambda_correlation
        self.min_split_gain = min_split_gain
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.main_task_strategy = main_task_strategy
        self.normalize_gradients = normalize_gradients
        self.loss = loss
        self.random_state = random_state
        self.verbose = verbose
        
        self.trees = []
        self.initial_predictions = None
        self.n_tasks = None
        self.feature_importances_ = None
        
        self._validate_parameters()
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _validate_parameters(self) -> None:
        """
        パラメータの検証
        """
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than 0")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        
        if self.max_depth <= 0:
            raise ValueError("max_depth must be greater than 0")
        
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        
        if self.gamma <= 0:
            raise ValueError("gamma must be greater than 0")
        
        if self.lambda_l2 < 0:
            raise ValueError("lambda_l2 must be non-negative")
        
        if self.lambda_correlation < 0:
            raise ValueError("lambda_correlation must be non-negative")
        
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        
        if not 0.0 < self.colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be in (0, 1]")
        
        if self.main_task_strategy not in ['random', 'rotate', 'fixed']:
            raise ValueError("main_task_strategy must be 'random', 'rotate', or 'fixed'")
        
        if isinstance(self.loss, str) and self.loss not in ["mse", "logloss"]:
            raise ValueError("loss must be 'mse', 'logloss', or a callable")
    
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
        if isinstance(self.loss, str):
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
        elif callable(self.loss):
            # カスタム損失関数の場合は平均値を使用
            return np.mean(y_multi, axis=0)
        else:
            raise ValueError("Invalid loss type")
    
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
        if isinstance(self.loss, str):
            if self.loss == "mse":
                # 二乗誤差損失の勾配とヘシアン
                # 勾配: 2 * (予測値 - 真の値)
                # ヘシアン: 2
                gradients = 2 * (y_pred - y_multi)
                hessians = np.ones_like(gradients) * 2
            elif self.loss == "logloss":
                # クロスエントロピー損失の勾配とヘシアン（二値分類）
                # シグモイド関数で確率に変換
                probs = 1 / (1 + np.exp(-y_pred))
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
        elif callable(self.loss):
            # カスタム損失関数
            gradients, hessians = self.loss(y_multi, y_pred)
        else:
            raise ValueError("Invalid loss type")
        
        return gradients, hessians
    
    def _normalize_gradients_hessians(self, gradients: np.ndarray, hessians: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        勾配とヘシアンを標準化
        
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
        if not self.normalize_gradients:
            return gradients, hessians
        
        n_samples, n_tasks = gradients.shape
        normalized_gradients = np.zeros_like(gradients)
        normalized_hessians = np.zeros_like(hessians)
        
        # 各タスクごとに標準化
        for task_idx in range(n_tasks):
            # 勾配の標準化
            grad_mean = np.mean(gradients[:, task_idx])
            grad_std = np.std(gradients[:, task_idx])
            
            if grad_std > 1e-8:  # ゼロ除算を防ぐ
                normalized_gradients[:, task_idx] = (gradients[:, task_idx] - grad_mean) / grad_std
            else:
                normalized_gradients[:, task_idx] = gradients[:, task_idx] - grad_mean
            
            # ヘシアンの標準化
            hess_mean = np.mean(hessians[:, task_idx])
            hess_std = np.std(hessians[:, task_idx])
            
            if hess_std > 1e-8:
                normalized_hessians[:, task_idx] = (hessians[:, task_idx] - hess_mean) / hess_std
            else:
                normalized_hessians[:, task_idx] = hessians[:, task_idx] - hess_mean
        
        return normalized_gradients, normalized_hessians
    
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
        if gradients.shape[1] <= 1:
            return np.eye(gradients.shape[1])
        
        # 勾配ベクトル間の相関を計算
        corr = np.corrcoef(gradients.T)
        
        # NaNを0に置き換え
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        
        return corr
    
    def _select_main_task(self, iteration: int, n_tasks: int) -> int:
        """
        メインタスクを選択
        
        Parameters:
        -----------
        iteration : int
            現在の反復回数
        n_tasks : int
            タスク数
            
        Returns:
        --------
        main_task_idx : int
            メインタスクのインデックス
        """
        if n_tasks <= 1:
            return 0
        
        if self.main_task_strategy == 'random':
            # ランダムに選択
            return np.random.randint(0, n_tasks)
        elif self.main_task_strategy == 'rotate':
            # 順番に選択
            return iteration % n_tasks
        elif self.main_task_strategy == 'fixed':
            # 常に最初のタスク
            return 0
        else:
            # デフォルトはランダム
            return np.random.randint(0, n_tasks)
    
    def fit(self, X: np.ndarray, y_multi: np.ndarray) -> 'MTGBM':
        """
        マルチタスクデータでモデルを学習
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクのターゲット値
            
        Returns:
        --------
        self : MTGBM
            学習済みモデル
        """
        # 入力検証
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y_multi.ndim == 1:
            y_multi = y_multi.reshape(-1, 1)
        
        if X.shape[0] != y_multi.shape[0]:
            raise ValueError(f"X and y_multi have different number of samples: {X.shape[0]} vs {y_multi.shape[0]}")
        
        self.n_tasks = y_multi.shape[1]
        n_features = X.shape[1]
        
        # 初期予測値を計算
        self.initial_predictions = self._compute_initial_predictions(y_multi)
        
        # 現在の予測値を初期化
        y_pred = np.tile(self.initial_predictions, (X.shape[0], 1))
        
        # 特徴量重要度を初期化
        self.feature_importances_ = np.zeros(n_features)
        
        # 学習開始時間
        start_time = time.time()
        
        # 各反復で木を構築
        for i in range(self.n_estimators):
            # 勾配とヘシアンを計算
            gradients, hessians = self._compute_gradients_hessians(y_multi, y_pred)
            
            # 勾配とヘシアンの標準化（オプション）
            gradients, hessians = self._normalize_gradients_hessians(gradients, hessians)
            
            # タスク間の相関を計算
            task_correlations = self._compute_task_correlations(gradients)
            
            # メインタスクを選択
            main_task_idx = self._select_main_task(i, self.n_tasks)
            
            # サブサンプリング
            sample_indices = np.arange(X.shape[0])
            if self.subsample < 1.0:
                n_samples_sub = max(int(X.shape[0] * self.subsample), 1)
                sample_indices = np.random.choice(X.shape[0], size=n_samples_sub, replace=False)
            
            X_subset = X[sample_indices]
            gradients_subset = gradients[sample_indices]
            hessians_subset = hessians[sample_indices]
            
            # 特徴量のサブサンプリング
            feature_indices = np.arange(n_features)
            if self.colsample_bytree < 1.0:
                n_features_sub = max(int(n_features * self.colsample_bytree), 1)
                feature_indices = np.random.choice(n_features, size=n_features_sub, replace=False)
            
            X_subset_features = X_subset[:, feature_indices]
            
            # 新しい木を構築
            tree = MultiTaskDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                gamma=self.gamma,
                lambda_l2=self.lambda_l2,
                lambda_correlation=self.lambda_correlation,
                min_split_gain=self.min_split_gain,
                random_state=self.random_state + i if self.random_state is not None else None
            )
            
            # 木を学習
            tree.fit(
                X_subset_features, 
                gradients_subset, 
                hessians_subset,
                task_correlations=task_correlations,
                main_task_idx=main_task_idx
            )
            
            # 木を保存
            self.trees.append({
                'tree': tree,
                'feature_indices': feature_indices
            })
            
            # 予測値を更新
            update = np.zeros_like(y_pred)
            update[sample_indices] = tree.predict(X_subset_features)
            y_pred += self.learning_rate * update
            
            # 特徴量重要度を更新
            self._update_feature_importance(tree, feature_indices)
            
            # 進捗を表示（オプション）
            if self.verbose >= 1 and ((i + 1) % 10 == 0 or i == 0 or i == self.n_estimators - 1):
                elapsed_time = time.time() - start_time
                
                # 評価指標を計算
                if isinstance(self.loss, str) and self.loss == "mse":
                    metric_value = np.mean((y_multi - y_pred) ** 2)
                    metric_name = "MSE"
                elif isinstance(self.loss, str) and self.loss == "logloss":
                    probs = 1 / (1 + np.exp(-y_pred))
                    probs = np.clip(probs, 1e-7, 1 - 1e-7)
                    metric_value = -np.mean(y_multi * np.log(probs) + (1 - y_multi) * np.log(1 - probs))
                    metric_name = "LogLoss"
                else:
                    metric_value = 0.0
                    metric_name = "Custom"
                
                print(f"Iter {i+1}/{self.n_estimators}, {metric_name}: {metric_value:.6f}, Time: {elapsed_time:.2f}s")
        
        # 特徴量重要度を正規化
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
        
        return self
    
    def _update_feature_importance(self, tree: MultiTaskDecisionTree, feature_indices: np.ndarray) -> None:
        """
        特徴量重要度を更新
        
        Parameters:
        -----------
        tree : MultiTaskDecisionTree
            学習済みの木
        feature_indices : array-like
            使用した特徴量のインデックス
        """
        def collect_gains(node, current_feature_indices):
            if node is None or node.is_leaf:
                return
            
            # 木のローカル特徴インデックスを元の特徴インデックスにマッピング
            original_feature_idx = current_feature_indices[node.feature_idx]
            self.feature_importances_[original_feature_idx] += node.split_gain
            
            collect_gains(node.left, current_feature_indices)
            collect_gains(node.right, current_feature_indices)
        
        collect_gains(tree.root, feature_indices)
    
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
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if not self.trees:
            raise ValueError("Model has not been trained yet")
        
        # 初期予測値
        y_pred = np.tile(self.initial_predictions, (X.shape[0], 1))
        
        # 各木の予測を加算
        for tree_info in self.trees:
            tree = tree_info['tree']
            feature_indices = tree_info['feature_indices']
            
            # 特徴量サブセットを使用
            X_subset = X[:, feature_indices]
            tree_prediction = tree.predict(X_subset)
            
            y_pred += self.learning_rate * tree_prediction
        
        # 損失関数に応じた後処理
        if isinstance(self.loss, str) and self.loss == "logloss":
            # ロジスティック回帰の場合：シグモイド関数で確率に変換
            y_pred = 1 / (1 + np.exp(-y_pred))
            # 数値安定性のためクリッピング
            y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        return y_pred
    
    def get_feature_importance(self) -> np.ndarray:
        """
        特徴量の重要度を取得
        
        Returns:
        --------
        feature_importance : array-like
            各特徴量の重要度
        """
        if self.feature_importances_ is None:
            raise ValueError("Model has not been trained yet")
        
        return self.feature_importances_
    
    def evaluate(self, X: np.ndarray, y_multi: np.ndarray, metrics: List[Union[str, Callable]] = ['mse']) -> Dict:
        """
        モデルの評価
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクの真のターゲット値
        metrics : list of str or callable, default=['mse']
            使用する評価指標のリスト
            
        Returns:
        --------
        results : dict
            各タスクの各評価指標の値
        """
        # 予測
        y_pred = self.predict(X)
        
        # 入力検証
        if y_multi.ndim == 1:
            y_multi = y_multi.reshape(-1, 1)
        
        # 結果格納用辞書
        results = {}
        
        # 各評価指標を計算
        for metric_func in metrics:
            if isinstance(metric_func, str):
                metric_name = metric_func.lower()
                
                if metric_name == 'mse':
                    # 平均二乗誤差
                    val = np.mean((y_multi - y_pred) ** 2, axis=0)
                    results['mse'] = val.tolist()
                    results['mse_avg'] = np.mean(val).item()
                
                elif metric_name == 'rmse':
                    # 平方根平均二乗誤差
                    val = np.sqrt(np.mean((y_multi - y_pred) ** 2, axis=0))
                    results['rmse'] = val.tolist()
                    results['rmse_avg'] = np.mean(val).item()
                
                elif metric_name == 'mae':
                    # 平均絶対誤差
                    val = np.mean(np.abs(y_multi - y_pred), axis=0)
                    results['mae'] = val.tolist()
                    results['mae_avg'] = np.mean(val).item()
                
                elif metric_name == 'r2':
                    # 決定係数
                    ss_res = np.sum((y_multi - y_pred) ** 2, axis=0)
                    ss_tot = np.sum((y_multi - np.mean(y_multi, axis=0, keepdims=True)) ** 2, axis=0)
                    
                    # ゼロ除算を防ぐ
                    r2_scores = np.zeros_like(ss_tot, dtype=float)
                    valid_mask = ss_tot > 1e-9
                    r2_scores[valid_mask] = 1 - (ss_res[valid_mask] / ss_tot[valid_mask])
                    r2_scores[~valid_mask & (ss_res <= 1e-9)] = 1.0  # 完全一致
                    r2_scores[~valid_mask & (ss_res > 1e-9)] = 0.0  # 定数モデルより悪い
                    
                    results['r2'] = r2_scores.tolist()
                    results['r2_avg'] = np.mean(r2_scores).item()
                
                elif metric_name == 'accuracy' and isinstance(self.loss, str) and self.loss == 'logloss':
                    # 二値分類の正解率
                    y_pred_binary = (y_pred > 0.5).astype(int)
                    val = np.mean(y_pred_binary == y_multi, axis=0)
                    results['accuracy'] = val.tolist()
                    results['accuracy_avg'] = np.mean(val).item()
                
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")
            
            elif callable(metric_func):
                # カスタム評価指標
                metric_name = getattr(metric_func, '__name__', 'custom_metric')
                val = metric_func(y_multi, y_pred)
                results[metric_name] = val
            
            else:
                raise ValueError(f"Invalid metric type: {type(metric_func)}")
        
        return results
