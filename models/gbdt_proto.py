import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import time
import json
from datetime import datetime
from sklearn.model_selection import KFold
from .base import MTGBMBase
from .gbdt_components.gradient_computer import GradientComputer
from .gbdt_components.weighting_strategies import WeightingStrategyManager
from .gbdt_components.data_transforms import _validate_strategy_n_tasks


def _add_cvr_labels(y: np.ndarray) -> np.ndarray:
    """
    [Click, CV] の2値ラベルを3タスク学習用の [CTR, CTCVR, CVR] 形式に変換します。

    パラメータ:
    -----------
    y : array-like, shape=(n_samples, 2)
        2値ラベル [Click, CV] を指定
        - y[:, 0] = Click (0 または 1)
        - y[:, 1] = CV (0 または 1)

    戻り値:
    -------
    y_converted : array-like, shape=(n_samples, 3)
        変換後のラベル [CTR, CTCVR, CVR]
        - y_converted[:, 0] = CTR = Click
        - y_converted[:, 1] = CTCVR = Click * CV
        - y_converted[:, 2] = CVR = Click=1 の場合 CV, それ以外は NaN
    """
    if y.shape[1] != 2:
        raise ValueError(f"Input y must have 2 columns [Click, CV], got {y.shape[1]}")
    
    n_samples = y.shape[0]
    y_converted = np.zeros((n_samples, 3))
    
    # CTR = Click
    y_converted[:, 0] = y[:, 0]
    
    # CTCVR = Click * CV
    y_converted[:, 1] = y[:, 0] * y[:, 1]
    
    # CVR = CV if Click=1, else NaN
    y_converted[:, 2] = np.where(y[:, 0] == 1, y[:, 1], np.nan)
    
    return y_converted


def _compute_ips_weight(ctr_pred: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    """
    Compute Inverse Propensity Score (IPS) weights for CVR task.
    
    Parameters:
    -----------
    ctr_pred : array-like, shape=(n_samples,)
        Predicted CTR probabilities
    epsilon : float, default=1e-8
        Small value to avoid division by zero
    
    Returns:
    --------
    ips_weights : array-like, shape=(n_samples,)
        IPS weights = 1 / max(ctr_pred, epsilon)
    """
    return 1.0 / np.maximum(ctr_pred, epsilon)


def _convert_to_ordinal_labels(y_click_cv: np.ndarray) -> np.ndarray:
    """
    2列 (click, cv) → 1列 (ordinal) への変換
    0: no click, no cv
    1: click only
    2: click+cv
    (click=0, cv=1)は理論上あり得ない前提
    """
    y_click = y_click_cv[:, 0]
    y_cv = y_click_cv[:, 1]
    ordinal = np.zeros_like(y_click, dtype=int)
    ordinal[(y_click == 1) & (y_cv == 0)] = 1
    ordinal[(y_click == 1) & (y_cv == 1)] = 2
    return ordinal.reshape(-1, 1)


class DecisionTreeNode:
    """
    決定木のノードクラス: ノードごとの処理機構を定義
    
    Attributes:
    -----------
    feature_idx : int or None
        分割に使用する特徴のインデックス(リーフノードの場合はNone)
    threshold : float or None
        分割の閾値(リーフノードの場合はNone)
    left : DecisionTreeNode or None
        左の子ノード
    right : DecisionTreeNode or None
        右の子ノード
    is_leaf : bool
        リーフノードかどうか
    values : array-like, shape=(n_tasks,)
        リーフノードの場合の予測値(各タスクごと)
    node_id : int
        ノードID
    depth : int
        ノードの深さ
    n_samples : int
        このノードのサンプル数
    ctr_rate : float
        このノードでのCTR率(タスク0の平均値)
    information_gain : float
        分割による情報利得(分割ノードの場合)
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
        self.ctr_rate = 0.0
        self.information_gain = 0.0
        
    def predict(self, X: np.ndarray, n_tasks: int) -> np.ndarray:
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
        
        predictions = np.zeros((X.shape[0], n_tasks))
        if np.any(left_mask):
            left_pred = self.left.predict(X[left_mask], n_tasks)
            predictions[left_mask] = left_pred
        if np.any(right_mask):
            right_pred = self.right.predict(X[right_mask], n_tasks)
            predictions[right_mask] = right_pred
        return predictions


class MultiTaskDecisionTree:
    """
    マルチタスク決定木クラス: 複数のタスクを同時に学習・予測する1本の決定木モデルを定義する
    
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
                 min_samples_leaf: int = 100,
                 n_tasks: int = 2,
                 weighting_strategy: str = "mtgbm",
                 gain_threshold: float = 0.1,
                 track_split_gains: bool = True,
                 verbose_logging: bool = False,
                 is_dynamic_weight: bool = False,
                 gamma: float = 50.0,
                 delta: float = 0.5,
                 random_state: Optional[int] = None,
                 threshold_prop_ctcvr: float = 0.5,
                 threshold_prop_cvr: float = 0.5):
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
        n_tasks : int, default=2
            タスク数
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
            勾配強調の倍率を表すハイパーパラメータ
        delta : float, default=0.5
            動的重み切り替えの情報利得閾値
        random_state : int, optional
            乱数シード
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_tasks = n_tasks
        self.weighting_strategy = weighting_strategy
        self.gain_threshold = gain_threshold
        self.track_split_gains = track_split_gains
        self.verbose_logging = verbose_logging
        self.is_dynamic_weight = is_dynamic_weight
        self.gamma = gamma
        self.delta = delta
        self.root = None
        self.threshold_prop_ctcvr = threshold_prop_ctcvr
        self.threshold_prop_cvr = threshold_prop_cvr
        
        # 分割利得の記録
        self.split_gains = []
        self.current_iteration = 0
        self.change_div_nodes = 0
        self.all_div_nodes = 0

        # ノード情報のログ
        self.node_logs = []
        self.node_counter = 0
        
        # 勾配重み
        self.gradient_weights = None
        if self.weighting_strategy == "adaptive_hybrid":
            # CTR:10, CTCVR:10, CVR:1
            self.gradient_weights = np.array([10.0, 10.0, 1.0])
        
        # mtgbm戦略: 木ごとにランダムでタスクを選択
        self.selected_task = None
        if self.weighting_strategy == "mtgbm":
            if random_state is not None:
                np.random.seed(random_state)
            # CTRとCTCVRの2つから選択(タスク0またはタスク1)
            self.selected_task = np.random.choice([0, 1])
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, 
            X: np.ndarray, 
            raw_gradients: np.ndarray, 
            raw_hessians: np.ndarray,
            split_gradients: np.ndarray,
            split_hessians: np.ndarray,
            y_true: Optional[np.ndarray] = None,
            iteration: int = 0,
            current_predictions: Optional[np.ndarray] = None) -> 'MultiTaskDecisionTree':
        """
        マルチタスク決定木を学習
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        raw_gradients : array-like, shape=(n_samples, n_tasks)
            予測値計算用の生の勾配
        raw_hessians : array-like, shape=(n_samples, n_tasks)
            予測値計算用の生のヘシアン
        split_gradients : array-like, shape=(n_samples, n_tasks)
            分割探索用の重み付けされた勾配
        split_hessians : array-like, shape=(n_samples, n_tasks)
            分割探索用の重み付けされたヘシアン
        y_true : array-like, shape=(n_samples, n_tasks), optional
            実際のラベル値(CVR率計算用)
        current_predictions : array-like, shape=(n_samples, n_tasks), optional
            現在のアンサンブル予測値(新戦略用)
            
        Returns:
        --------
        self : MultiTaskDecisionTree
            学習済みモデル
        """
        self.n_tasks = raw_gradients.shape[1]
        self.raw_gradients = raw_gradients
        self.raw_hessians = raw_hessians
        self.split_gradients = split_gradients
        self.split_hessians = split_hessians
        
        self.root = DecisionTreeNode()
        self._build_tree(
            self.root, X, np.arange(X.shape[0]),
            depth=0, y_true=y_true, current_predictions=current_predictions
        )
        # print(f"タスク変更回数記録: {[self.change_div_nodes, self.all_div_nodes]}")
        return self
    
    def _compute_ensemble_weights_gradients_hessians(self, 
                                                    gradients: np.ndarray, 
                                                    hessians: np.ndarray,
                                                    task_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        アンサンブル勾配・ヘシアン計算(アルゴリズム2)，正規化処理を含む
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配(タスク0: CTR, タスク1: CTCVR)
        hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのヘシアン
        task_weights : array-like, shape=(n_tasks,), optional
            動的に決定されたタスク重み。指定されない場合はデフォルトの重みを使用。
            
        Returns:
        --------
        ensemble_gradients : array-like, shape=(n_samples,)
            アンサンブル勾配
        ensemble_hessians : array-like, shape=(n_samples,)
            アンサンブルヘシアン
        """
        n_samples, n_tasks = gradients.shape
        # 3タスクモードでCVRタスクにNaN値がある場合はマスクを作成
        mask = None
        if gradients.shape[1] == 3:
            mask = ~np.isnan(gradients)
        
        # 正規化処理を行わない戦略はそのまま返す
        if self.weighting_strategy == "ctcvr_subctr_de_norm" or self.weighting_strategy == "mtgbm-de-norm":
            normalized_gradients = gradients
            normalized_hessians = hessians
        else:
            # 1. 勾配の正規化計算(平均0.05、標準偏差0.01)
            normalized_gradients = self._compute_normalization_weights(gradients, target_mean=0.05, target_std=0.01, mask=mask)
            # 2. ヘシアンの正規化計算(平均1.0、標準偏差0.1)
            normalized_hessians = self._compute_normalization_weights(hessians, target_mean=1.00, target_std=0.1, mask=mask)
        # 3. 勾配とヘシアンの加重和を計算
        if task_weights is not None:
            weights = task_weights
        elif self.weighting_strategy == 'ctcvr-subctr' and n_tasks == 3:
            # CTCVR:CTR=10:1, CVR=0
            weights = np.array([1.0, 10.0, 0.0])
        elif self.gradient_weights is not None:
            weights = self.gradient_weights
        else:
            # デフォルトは均等重み
            weights = np.ones(n_tasks) / n_tasks
        ensemble_gradients = np.sum(normalized_gradients * weights, axis=1)
        ensemble_hessians = np.sum(normalized_hessians * weights, axis=1)
        
        return ensemble_gradients, ensemble_hessians
    
    def _compute_ensemble_weights_gradients_hessians_afterIPS(self, 
                                                            gradients: np.ndarray, 
                                                            hessians: np.ndarray,
                                                            probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        mtgbm-three-afterIPS戦略用: 正規化後にIPS重みづけを適用するアンサンブル勾配・ヘシアン計算
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのヘシアン
        probs : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの予測確率
            
        Returns:
        --------
        ensemble_gradients : array-like, shape=(n_samples,)
            アンサンブル勾配(IPS重みづけ後)
        ensemble_hessians : array-like, shape=(n_samples,)
            アンサンブルヘシアン(IPS重みづけ後)
        """
        n_samples, n_tasks = gradients.shape
        
        # 1. 3タスクモードでCVRタスクにNaN値がある場合はマスクを作成
        mask = None
        if gradients.shape[1] == 3:
            mask = ~np.isnan(gradients)
        
        # 2. 勾配・ヘシアンを正規化(IPS適用前)
        normalized_gradients = self._compute_normalization_weights(gradients, target_mean=0.05, target_std=0.01, mask=mask)
        normalized_hessians = self._compute_normalization_weights(hessians, target_mean=1.00, target_std=0.1, mask=mask)
        
        # 3. CVRタスクにIPS重みづけを適用(正規化後)
        if self.n_tasks == 3:
            # CTR予測値を取得(タスク0、確率に変換)
            ctr_pred = probs[:, 0]
            
            # IPS重みを計算
            ips_weights = _compute_ips_weight(ctr_pred)
            
            # CVRタスク(タスク2)の正規化済み勾配・ヘシアンにIPS重みを適用
            # Click=1のサンプルのみ有効(NaNでないサンプル)
            cvr_mask = ~np.isnan(gradients[:, 2])
            normalized_gradients[cvr_mask, 2] *= ips_weights[cvr_mask]
            normalized_hessians[cvr_mask, 2] *= ips_weights[cvr_mask]
            
            # NaNサンプルの勾配・ヘシアンを0にセット
            nan_mask = np.isnan(gradients[:, 2])
            normalized_gradients[nan_mask, 2] = 0.0
            normalized_hessians[nan_mask, 2] = 0.0
        
        # 4. 勾配とヘシアンの加重和を計算
        # gradient_weightsがない場合は均等重み
        if self.gradient_weights is None:
            weights = np.ones(n_tasks) / n_tasks
        else:
            weights = self.gradient_weights
        
        ensemble_gradients = np.sum(normalized_gradients * weights, axis=1)
        ensemble_hessians = np.sum(normalized_hessians * weights, axis=1)
        
        return ensemble_gradients, ensemble_hessians
    
    def _compute_normalization_weights(self, data: np.ndarray, target_mean: float, target_std: float, mask: Optional[np.ndarray] = None) -> np.ndarray:
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
        mask : array-like, shape=(n_samples, n_tasks), optional
            有効なデータを示すマスク配列(TrueまたはFalseで構成)
            NaN値がある場合(CVRタスクなど)に使用
            
        Returns:
        --------
        normalized_data : array-like, shape=(n_samples, n_tasks)
            正規化後のデータ
        """
        n_samples, n_tasks = data.shape
        normalized_data = np.zeros_like(data)
        
        for task_idx in range(n_tasks):
            task_data = data[:, task_idx]
            
            # マスクが指定されている場合は有効データのみで統計量を計算
            if mask is not None:
                task_mask = mask[:, task_idx]
                valid_data = task_data[task_mask]
                
                if len(valid_data) == 0:
                    # 有効データがない場合は元のデータをそのまま保持
                    normalized_data[:, task_idx] = task_data.copy()
                    continue
                    
                current_mean = np.mean(valid_data)
                current_std = np.std(valid_data)
                
                if current_std == 0:
                    # 標準偏差が0の場合は目標平均値への単純シフト(有効データのみ)
                    normalized_data[task_mask, task_idx] = target_mean
                    # 無効データは元の値を保持(NaNなど)
                    normalized_data[~task_mask, task_idx] = task_data[~task_mask]
                else:
                    # 線形変換: (data - current_mean) / current_std * target_std + target_mean
                    normalized_data[task_mask, task_idx] = (valid_data - current_mean) / current_std * target_std + target_mean
                    # 無効データは元の値を保持(NaNなど)
                    normalized_data[~task_mask, task_idx] = task_data[~task_mask]
            else:
                # マスクが指定されていない場合は従来通りの処理
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
                indices: np.ndarray,
                depth: int,
                y_true: Optional[np.ndarray] = None,
                current_predictions: Optional[np.ndarray] = None):
        """
        再帰的に決定木を構築(アルゴリズム1)
        
        Parameters:
        -----------
        node : DecisionTreeNode
            現在のノード
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        indices : array-like, shape=(n_node_samples,)
            現在のノードに含まれるサンプルのインデックス
        depth : int
            現在の深さ
        y_true : array-like, shape=(n_samples, n_tasks), optional
            実際のラベル値
        current_predictions : array-like, shape=(n_samples, n_tasks), optional
            現在の予測値
        """
        n_node_samples = len(indices)
        node.n_samples = n_node_samples
        node.depth = depth
        if y_true is not None:
            node.ctr_rate = np.mean(y_true[indices, 0]) if n_node_samples > 0 else 0.0

        # 停止条件
        is_leaf_node = (
            depth >= self.max_depth or
            n_node_samples < self.min_samples_split or
            (y_true is not None and len(np.unique(y_true[indices, 0])) == 1)
        )

        # 最適な分割を見つける
        best_split = self._find_best_split(X, indices, y_true)

        # best_splitがNoneならリーフノードにする
        if is_leaf_node or best_split is None:
            node.is_leaf = True
            node.values = self._calculate_leaf_value(indices)
            return

        # ノードを分割情報で更新
        node.feature_idx = best_split['feature_idx']
        node.threshold = best_split['threshold']
        node.information_gain = best_split['gain']

        left_indices = best_split['left_indices']
        right_indices = best_split['right_indices']

        self.node_counter += 1
        node.left = DecisionTreeNode(node_id=self.node_counter, depth=depth + 1)
        self._build_tree(node.left, X, left_indices, depth + 1, y_true, current_predictions)

        self.node_counter += 1
        node.right = DecisionTreeNode(node_id=self.node_counter, depth=depth + 1)
        self._build_tree(node.right, X, right_indices, depth + 1, y_true, current_predictions)
    
    def _calculate_leaf_value(self, indices: np.ndarray) -> np.ndarray:
        """
        リーフノードの値を計算
        mtgbm戦略の場合はselected_taskのみ生の勾配・ヘシアンからリーフ値を計算し、他タスクは0にする。
        adaptive_hybrid戦略の場合はCTCVRリーフ値をCTRリーフ値×CVRリーフ値で計算する。
        Parameters:
        -----------
        indices : array-like, shape=(n_leaf_samples,)
            リーフノードに含まれるサンプルのインデックス
        Returns:
        --------
        leaf_values : array-like, shape=(n_tasks,)
            各タスクのリーフ値
        """
        leaf_values = np.zeros(self.n_tasks)
        gradients_subset = self.raw_gradients[indices]
        hessians_subset = self.raw_hessians[indices]

        if self.weighting_strategy == "mtgbm" and self.selected_task is not None:
            # selected_task以外も通常通り計算
            for task_idx in range(self.n_tasks):
                grad = gradients_subset[:, task_idx]
                hess = hessians_subset[:, task_idx]
                if np.any(np.isnan(grad)):
                    valid_mask = ~np.isnan(grad)
                    grad = grad[valid_mask]
                    hess = hess[valid_mask]
                sum_grad = np.sum(grad)
                sum_hess = np.sum(hess)
                if sum_hess == 0:
                    leaf_values[task_idx] = 0.0
                else:
                    leaf_values[task_idx] = -sum_grad / sum_hess
        elif self.weighting_strategy == "adaptive_hybrid" and self.n_tasks == 3:
            # print(f"[adaptive_hybrid] leaf node sample size: {len(indices)}")
            # CTR
            grad_ctr = gradients_subset[:, 0]
            hess_ctr = hessians_subset[:, 0]
            ctr_value = -np.sum(grad_ctr) / np.sum(hess_ctr) if np.sum(hess_ctr) != 0 else 0.0

            # CVR
            grad_cvr = gradients_subset[:, 2]
            hess_cvr = hessians_subset[:, 2]
            valid_mask = ~np.isnan(grad_cvr)
            grad_cvr = grad_cvr[valid_mask]
            hess_cvr = hess_cvr[valid_mask]
            cvr_value = -np.sum(grad_cvr) / np.sum(hess_cvr) if np.sum(hess_cvr) != 0 else 0.0

            # CTCVR = CTR × CVR
            ct_cvr_value = ctr_value * cvr_value

            leaf_values[0] = ctr_value
            leaf_values[1] = ct_cvr_value
            leaf_values[2] = cvr_value
        else:
            # 従来通り全タスクで計算
            for task_idx in range(self.n_tasks):
                grad = gradients_subset[:, task_idx]
                hess = hessians_subset[:, task_idx]
                if np.any(np.isnan(grad)):
                    valid_mask = ~np.isnan(grad)
                    grad = grad[valid_mask]
                    hess = hess[valid_mask]
                sum_grad = np.sum(grad)
                sum_hess = np.sum(hess)
                if sum_hess == 0:
                    leaf_values[task_idx] = 0.0
                else:
                    leaf_values[task_idx] = -sum_grad / sum_hess
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
                
                # 正則化項(XGBoostのデフォルトに合わせて小さく設定)
                lambda_reg = 0.01  # L2正則化パラメータ
                gamma_reg = 0.0   # 複雑性に対するペナルティ
                
                # 情報利得を計算(XGBoostの公式を使用)
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

    def _find_best_split(self, X: np.ndarray, indices: np.ndarray, y_true: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """
        最適な分割を見つける
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        indices : array-like, shape=(n_node_samples,)
            現在のノードに含まれるサンプルのインデックス
        y_true : array-like, shape=(n_samples, n_tasks), optional
            実際のラベル値(adaptive_hybrid戦略用)
            
        Returns:
        --------
        best_split : dict or None
            最適な分割情報(特徴インデックス、閾値、情報利得、左右のインデックス)
            見つからない場合はNone
        """
        n_node_samples = len(indices)
        if n_node_samples < self.min_samples_split:
            return None

        X_node = X[indices]
        # --- 既存分岐・勾配準備を活かし、分割探索直前でctcvr_subctr_de_norm_gain戦略のCTCVR→CTR切り替え ---
        gradients_node = None
        hessians_node = None
        task_weights = None
        # ...既存の分岐（adaptive_hybrid, ctcvr-subctr, mtgbm, ablation_STGBDT_normalize, etc.）...
        # ...（この部分は既存コードをそのまま流用）...
        # 他の戦略では分割探索用の勾配とヘシアンを使用
        if gradients_node is None or hessians_node is None:
            gradients_node = self.split_gradients[indices]
            hessians_node = self.split_hessians[indices]

        # --- 分割探索直前で新戦略の切り替え ---
        if self.weighting_strategy == 'ctcvr_subctr_de_norm_gain':
            self.all_div_nodes += 1
            # まずCTCVRタスク（タスク1）で分割探索
            ensemble_gradients = gradients_node[:, 1]
            ensemble_hessians = hessians_node[:, 1]
            best_gain, best_feature_idx, best_threshold, left_indices_local, right_indices_local = self._search_best_split(
                X_node, ensemble_gradients, ensemble_hessians
            )
            # gainがself.delta未満ならCTRタスク（タスク0）で再探索
            if best_gain < self.delta:
                # print("CTR利用")
                ensemble_gradients = gradients_node[:, 0]
                ensemble_hessians = hessians_node[:, 0]
                best_gain, best_feature_idx, best_threshold, left_indices_local, right_indices_local = self._search_best_split(
                    X_node, ensemble_gradients, ensemble_hessians
                )
                self.change_div_nodes += 1
            

            if best_gain > 0:
                return {
                    'gain': best_gain,
                    'feature_idx': best_feature_idx,
                    'threshold': best_threshold,
                    'left_indices': indices[left_indices_local],
                    'right_indices': indices[right_indices_local]
                }
            return None
        
        # --- B: シングルタスクGBDTは正規化・アンサンブル処理を完全バイパス ---
        if self.n_tasks == 1:
            # raw_gradientsとraw_hessians: 正規化・重みづけなし
            ensemble_gradients = self.raw_gradients[indices].flatten()
            ensemble_hessians = self.raw_hessians[indices].flatten()
        else:
            task_weights = None
            # --- A: adaptive_hybrid戦略: 重み分岐をraw_gradientsの負の割合で判定 ---
            if self.weighting_strategy == 'adaptive_hybrid':
                raw_gradients_node = self.raw_gradients[indices]

                # CTCVRタスクの負の勾配割合
                prop_neg_ctcvr = np.mean(raw_gradients_node[:, 1] < 0)
                # CVRタスクの負の勾配割合(NaN除外)
                cvr_gradients = raw_gradients_node[:, 2]
                valid_cvr_mask = ~np.isnan(cvr_gradients)
                if np.any(valid_cvr_mask):
                    prop_neg_cvr = np.mean(cvr_gradients[valid_cvr_mask] < 0)
                else:
                    prop_neg_cvr = 0.0

                # CTCVRとCVRの負の勾配割合が閾値を超えた場合の切り替えフラグ
                use_ctcvr = prop_neg_ctcvr >= self.threshold_prop_ctcvr
                use_cvr = prop_neg_cvr >= self.threshold_prop_cvr

                # デフォルト重み(10, 1, 10)を使用し、除外タスクの重みを0にする
                w_ctr = 10.0  # CTRは常に使用
                w_ctcvr = 10.0 if use_ctcvr else 0.0  # CTCVR: デフォルト10、除外時0
                w_cvr = 1.0 if use_cvr else 0.0       # CVR: デフォルト1、除外時0
                task_weights = np.array([w_ctr, w_ctcvr, w_cvr])

                # adaptive_hybrid戦略でも、分岐探索にはIPS補正済み勾配を使用
                gradients_node = self.split_gradients[indices]
                hessians_node = self.split_hessians[indices]

            # --- B: ctcvr-subctr戦略 ---
            elif self.weighting_strategy == 'ctcvr-subctr':
                raw_gradients_node = self.raw_gradients[indices]
                prop_neg_ctcvr = np.mean(raw_gradients_node[:, 1] < 0)
                use_ctcvr = prop_neg_ctcvr >= self.threshold_prop_ctcvr
                # CTCVR:CTR=10:1, CVR=0, use_ctcvrがFalseならCTCVR=0
                w_ctr = 1.0
                w_ctcvr = 10.0 if use_ctcvr else 0.0
                w_cvr = 0.0
                task_weights = np.array([w_ctr, w_ctcvr, w_cvr])
                gradients_node = self.split_gradients[indices]
                hessians_node = self.split_hessians[indices]

            # --- C: mtgbm戦略でランダムタスク選択 --- --- C2: mtgbm-de-norm戦略（正規化なし） ---
            elif self.weighting_strategy == 'mtgbm' or self.weighting_strategy == 'mtgbm-de-norm':
                # 木ごとにランダム選択されたタスクのみ重み1、他は0
                if self.n_tasks >= 2:
                    task_weights = np.zeros(self.n_tasks)
                    task_weights[self.selected_task] = 1.0
                gradients_node = self.split_gradients[indices]
                hessians_node = self.split_hessians[indices]
            # --- D: ablation_STGBDT_normalize戦略 ---
            elif self.weighting_strategy == 'ablation_STGBDT_normalize':
                gradients_node = self.split_gradients[indices]
                hessians_node = self.split_hessians[indices]
                mask = None
                if gradients_node.shape[1] == 3:
                    mask = ~np.isnan(gradients_node)
                
                # ほんとはここで正規化してはいけないが，STGBDTでは正規化処理を含むアンサンブル処理を行わないので，ここで正規化する
                normalized_gradients = self._compute_normalization_weights(gradients_node, target_mean=0.05, target_std=0.01, mask=mask)
                normalized_hessians = self._compute_normalization_weights(hessians_node, target_mean=1.00, target_std=0.1, mask=mask)
                # 各タスクごとに分割探索（STGBDT同様）
                best_gain = 0.0
                best_feature_idx = None
                best_threshold = None
                best_left_indices = None
                best_right_indices = None
                for task_idx in range(self.n_tasks):
                    task_weights = np.zeros(self.n_tasks)
                    task_weights[task_idx] = 1.0
                    ensemble_gradients = np.sum(normalized_gradients * task_weights, axis=1)
                    ensemble_hessians = np.sum(normalized_hessians * task_weights, axis=1)
                    gain, feature_idx, threshold, left_indices_local, right_indices_local = self._search_best_split(
                        X_node, ensemble_gradients, ensemble_hessians
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_feature_idx = feature_idx
                        best_threshold = threshold
                        best_left_indices = left_indices_local
                        best_right_indices = right_indices_local
                if best_gain > 0:
                    return {
                        'gain': best_gain,
                        'feature_idx': best_feature_idx,
                        'threshold': best_threshold,
                        'left_indices': indices[best_left_indices],
                        'right_indices': indices[best_right_indices]
                    }
                return None
            else:
                # 他の戦略では分割探索用の勾配とヘシアンを使用
                gradients_node = self.split_gradients[indices]
                hessians_node = self.split_hessians[indices]

            # --- A, C: マルチタスク用アンサンブル勾配・ヘシアン計算 ---
            ensemble_gradients, ensemble_hessians = self._compute_ensemble_weights_gradients_hessians(
                gradients_node, hessians_node, task_weights
            )

        best_gain, best_feature_idx, best_threshold, left_indices_local, right_indices_local = self._search_best_split(
            X_node, ensemble_gradients, ensemble_hessians
        )

        if best_gain > 0:
            return {
                'gain': best_gain,
                'feature_idx': best_feature_idx,
                'threshold': best_threshold,
                'left_indices': indices[left_indices_local],
                'right_indices': indices[right_indices_local]
            }
        return None
    
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
            'ctr_rate': node.ctr_rate,
            'is_leaf': is_leaf,
            'feature_idx': node.feature_idx if not is_leaf else None,
            'threshold': node.threshold if not is_leaf else None,
            'information_gain': node.information_gain if not is_leaf else 0.0,
            'leaf_values': node.values.tolist() if hasattr(node.values, 'tolist') else node.values,
            'weight_switched': weight_switched,
        }
        
        self.node_logs.append(log_entry)
        
        # デバッグ用の詳細出力(オプション)
        if self.verbose_logging:
            if is_leaf:
                print(f"Leaf Node {node.node_id}: depth={node.depth}, samples={node.n_samples}, CVR={node.ctr_rate:.4f}")
            else:
                print(f"Split Node {node.node_id}: depth={node.depth}, samples={node.n_samples}, CVR={node.ctr_rate:.4f}, gain={node.information_gain:.4f}, feature={node.feature_idx}, threshold={node.threshold:.4f}, weight_switched={weight_switched}")
    
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
        n_tasks = self.n_tasks
        return self.root.predict(X, n_tasks)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        確率予測を行う(ログオッズ空間の予測値をシグモイド変換で確率に変換)
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        predictions : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの確率予測値
        """
        # ログオッズ空間の予測値を取得
        logits = self.predict(X)
        
        # シグモイド関数で確率に変換(数値安定性のためクリッピング)
        y_proba = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        
        # mtgbm-ctr-cvr戦略の場合、CTCVR = CTR × CVRで計算
        if (hasattr(self, 'weighting_strategy') and self.weighting_strategy == "mtgbm-ctr-cvr" and
            self.n_tasks == 3):
            # CTR(タスク0) × CVR(タスク2) = CTCVR(タスク1)
            y_proba[:, 1] = y_proba[:, 0] * y_proba[:, 2]
        
        return y_proba


class MTGBDT(MTGBMBase):
    """
    汎用マルチタスク勾配ブースティング決定木(MT-GBDT)のスクラッチ実装
    
    シングルタスク(n_tasks=1)とマルチタスク(n_tasks>=2)の両方に対応。
    シングルタスクの場合は従来のGBDTと同等の動作をし、
    マルチタスクの場合は各タスク間の相関を活用した学習を行う。
    
    Attributes:
    -----------
    n_estimators : int
        ブースティング反復回数(木の数)
    learning_rate : float
        学習率(各木の寄与度)
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
    trees_ : list of MultiTaskDecisionTree
        学習済みの木のリスト
    initial_predictions_ : array-like, shape=(n_tasks,)
        初期予測値
    """
    
    def __init__(self, 
        n_estimators: int = 100, 
        learning_rate: float = 0.1, 
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_tasks: int = 2,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        gradient_weights: Optional[np.ndarray] = None,
        normalize_gradients: bool = False,
        loss: str = "logloss",
        weighting_strategy: str = "mtgbm",
        gain_threshold: float = 0.1,
        track_split_gains: bool = True,
        is_dynamic_weight: bool = False,
        gamma: float = 50.0,
        delta: float = 0.5,
        verbose_logging: bool = False,
        random_state: Optional[int] = None,
        n_folds_oof: int = 5,
        threshold_prop_ctcvr: float = 0.5,
        threshold_prop_cvr: float = 0.5):
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        """
        マルチタスク勾配ブースティング決定木(MT-GBDT)の初期化
        Parameters:
        -----------
        n_estimators : int
            ブースティング反復回数(木の数)
        learning_rate : float
            学習率(各木の寄与度)
        max_depth : int
            各木の最大深さ
        min_samples_split : int
            分割に必要な最小サンプル数
        min_samples_leaf : int
            リーフノードに必要な最小サンプル数
        n_tasks : int
            タスク数(2以上でマルチタスク)
        subsample : float
            各木を構築する際のサンプリング率(0.0 < subsample <= 1.0)
        colsample_bytree : float
            各木を構築する際の特徴サンプリング率(0.0 < colsample_bytree <= 1.0)
        gradient_weights : np.ndarray, optional
            各タスクの勾配の重み(Noneの場合は均等重み)
        normalize_gradients : bool
            勾配を正規化するかどうか(Trueの場合、勾配を平均0、標準偏差1に正規化)
        loss : str
            損失関数("logloss" または "mse")
        weighting_strategy : str
            勾配の重み付け戦略("mtgbm", "adaptive_hybrid", "mtgbm-ctr-cvr" など)
        gain_threshold : float
            分割の情報利得の閾値(この値以下の分割は行わない)
        track_split_gains : bool
            分割ごとの情報利得を追跡するかどうか(Trueの場合、各分割の利得を記録)
        is_dynamic_weight : bool
            動的重み付けを使用するかどうか(Trueの場合、勾配の重みを動的に調整)
        gamma : float
            動的重み付けのパラメータ(デフォルトは50.0)
        delta : float
            動的重み付けのパラメータ(デフォルトは0.5)
        verbose_logging : bool
            ログ出力を詳細にするかどうか(Trueの場合、各ノードの詳細を出力)
        random_state : int, optional
            乱数シード(再現性のため)
        n_folds_oof : int
            OOF予測のためのフォールド数(adaptive_hybrid戦略用)
        threshold_prop_ctcvr : float
            CTCVRタスクの負の勾配割合の閾値(adaptive_hybrid戦略用)
        threshold_prop_cvr : float
            CVRタスクの負の勾配割合の閾値(adaptive_hybrid戦略用)
        """
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_tasks = n_tasks
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
        self.verbose_logging = verbose_logging

        self.n_folds_oof = n_folds_oof
        self.threshold_prop_ctcvr = threshold_prop_ctcvr
        self.threshold_prop_cvr = threshold_prop_cvr
        
        self.trees_ = []
        self.initial_predictions_ = None
        self.eval_results_ = {}
        self.best_iteration_ = 0
        self.best_score_ = float('inf')
        self.oof_ctr_preds_ = None
    
    def _compute_initial_predictions(self, y_multi: np.ndarray) -> np.ndarray:
        """
        初期予測値を計算(ログオッズ空間で計算、3タスク対応でCVRのNaN値を適切に処理)
        """
        if self.n_tasks == 3:
            initial_preds = np.zeros(self.n_tasks)

            ctr_prob = np.clip(np.mean(y_multi[:, 0]), 1e-7, 1-1e-7)
            initial_preds[0] = np.log(ctr_prob / (1 - ctr_prob))

            ctcvr_prob = np.clip(np.mean(y_multi[:, 1]), 1e-7, 1-1e-7)
            initial_preds[1] = np.log(ctcvr_prob / (1 - ctcvr_prob))

            cvr_mask = ~np.isnan(y_multi[:, 2])
            if np.sum(cvr_mask) > 0:
                cvr_prob = np.clip(np.mean(y_multi[cvr_mask, 2]), 1e-7, 1-1e-7)
                initial_preds[2] = np.log(cvr_prob / (1 - cvr_prob))
            else:
                initial_preds[2] = 0.0
                
            return initial_preds
        elif self.weighting_strategy == 'ordinal':
            # 2閾値の初期値（全体分布から）
            y_ord = _convert_to_ordinal_labels(y_multi)
            # theta1: P(y>=1), theta2: P(y>=2)
            p1 = np.clip(np.mean(y_ord >= 1), 1e-7, 1-1e-7)
            p2 = np.clip(np.mean(y_ord == 2), 1e-7, 1-1e-7)
            theta1 = np.log(p1 / (1 - p1))
            theta2 = np.log(p2 / (1 - p2))
            return np.array([theta1, theta2])
        else:
            initial_preds = np.zeros(self.n_tasks)
            for task in range(self.n_tasks):
                task_prob = np.clip(np.mean(y_multi[:, task]), 1e-7, 1-1e-7)
                initial_preds[task] = np.log(task_prob / (1 - task_prob))
            return initial_preds
    
    def _compute_raw_gradients_hessians(self, y_multi: np.ndarray, y_pred_logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        生の勾配とヘシアンを計算する。
        """
        if self.loss == "logloss":
            probs = 1 / (1 + np.exp(-np.clip(y_pred_logits, -500, 500)))
            gradients = probs - y_multi
            hessians = probs * (1 - probs)
            hessians = np.maximum(hessians, 1e-7)
            if self.n_tasks == 3:
                nan_mask = np.isnan(y_multi[:, 2])
                gradients[nan_mask, 2] = 0.0
                hessians[nan_mask, 2] = 0.0
        elif self.loss == "mse":
            gradients = 2 * (y_pred_logits - y_multi)
            hessians = np.ones_like(gradients) * 2
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")
        return gradients, hessians

    def _prepare_gradients_for_splitting(self, raw_gradients: np.ndarray, raw_hessians: np.ndarray, current_predictions: np.ndarray, iteration: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        分岐決定用の勾配・ヘシアンを準備(IPS補正など)
        """
        split_gradients = raw_gradients.copy()
        split_hessians = raw_hessians.copy()

        if self.n_tasks == 3:
            if self.weighting_strategy == 'adaptive_hybrid':
                if self.oof_ctr_preds_ is None:
                    raise ValueError("OOF CTR predictions are not generated for adaptive_hybrid strategy.")
                ips_weights = _compute_ips_weight(self.oof_ctr_preds_)
            else:
                probs = 1 / (1 + np.exp(-np.clip(current_predictions, -500, 500)))
                ctr_pred = probs[:, 0]
                ips_weights = _compute_ips_weight(ctr_pred)

            cvr_mask = ~np.isnan(split_gradients[:, 2])
            split_gradients[cvr_mask, 2] *= ips_weights[cvr_mask]
            split_hessians[cvr_mask, 2] *= ips_weights[cvr_mask]

        return split_gradients, split_hessians

    def _calculate_loss(self, y_true, y_pred_logits):
        if self.weighting_strategy == 'ordinal':
            y_ord = _convert_to_ordinal_labels(y_true)
            return self._ordinal_proportional_odds_loss(y_ord, y_pred_logits)
        elif self.loss == "logloss":
            y_pred_proba = 1 / (1 + np.exp(-np.clip(y_pred_logits, -500, 500)))
            y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
            
            if self.n_tasks == 3:
                logloss = - (y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
                return np.nanmean(logloss)
            else:
                return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
        elif self.loss == "mse":
            return np.mean((y_true - y_pred_logits) ** 2)
        else:
            raise ValueError(f"Unsupported loss: {self.loss}")

    def _generate_oof_ctr_predictions(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Out-of-Fold (OOF) を用いて、学習データに対するCTR予測値を生成します。
        """
        oof_preds = np.zeros(X.shape[0])
        kf = KFold(n_splits=self.n_folds_oof, shuffle=True, random_state=self.random_state)

        oof_model = MTGBDT(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_tasks=1,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            loss='logloss',
            weighting_strategy='stgbdt_baseline',
            random_state=self.random_state
        )

        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train_ctr = y[train_index, 0].reshape(-1, 1)

            oof_model.fit(X_train, y_train_ctr)
            oof_preds[val_index] = oof_model.predict_proba(X_val)[:, 0]

        return oof_preds

    def _ordinal_proportional_odds_loss(self, y_ord: np.ndarray, logits: np.ndarray) -> float:
        """
        順序ロジスティック回帰（比例オッズモデル）の損失
        y_ord: (n_samples, 1) 0,1,2
        logits: (n_samples, 2) 2つの閾値
        """
        # logits: shape (n_samples, 2)  (theta_1, theta_2)
        theta1 = logits[:, 0]
        theta2 = logits[:, 1]
        # P(y >= 1) = sigmoid(theta1), P(y >= 2) = sigmoid(theta2)
        p1 = 1 / (1 + np.exp(-theta1))
        p2 = 1 / (1 + np.exp(-theta2))
        # 各クラスの確率
        prob0 = 1 - p1
        prob1 = p1 - p2
        prob2 = p2
        # one-hot
        y0 = (y_ord[:, 0] == 0)
        y1 = (y_ord[:, 0] == 1)
        y2 = (y_ord[:, 0] == 2)
        eps = 1e-9
        loss = -np.mean(y0 * np.log(prob0 + eps) + y1 * np.log(prob1 + eps) + y2 * np.log(prob2 + eps))
        return loss

    def _ordinal_proportional_odds_grad_hess(self, y_ord: np.ndarray, logits: np.ndarray):
        """
        順序ロジスティック回帰の勾配・ヘッセ行列
        y_ord: (n_samples, 1)
        logits: (n_samples, 2)
        戻り値: grad, hess (shape: (n_samples, 2))
        """
        theta1 = logits[:, 0]
        theta2 = logits[:, 1]
        p1 = 1 / (1 + np.exp(-theta1))
        p2 = 1 / (1 + np.exp(-theta2))
        y0 = (y_ord[:, 0] == 0)
        y1 = (y_ord[:, 0] == 1)
        y2 = (y_ord[:, 0] == 2)
        grad1 = -y0 * p1 + y1 * (1 - p1) - y1 * p2 + y2 * (0)
        grad2 = -y1 * p2 + y2 * (1 - p2)
        grad = np.stack([grad1, grad2], axis=1)
        # ヘッセ行列（対角のみ）
        hess1 = y0 * p1 * (1 - p1) + y1 * p1 * (1 - p1)
        hess2 = y1 * p2 * (1 - p2) + y2 * p2 * (1 - p2)
        hess = np.stack([hess1, hess2], axis=1)
        hess = np.maximum(hess, 1e-7)
        return grad, hess

    def fit(self, X: np.ndarray, y_multi: np.ndarray, **kwargs) -> 'MTGBDT':
        """
        シングル/マルチタスクデータでモデルを学習
        """
        X, y_multi = self._validate_input(X, y_multi)

        self.trees_ = []
        self.eval_results_ = {}
        self.best_iteration_ = 0
        self.best_score_ = float('inf')
        self.oof_ctr_preds_ = None

        if self.weighting_strategy == 'adaptive_hybrid':
            if self.n_tasks != 3:
                raise ValueError("adaptive_hybrid strategy is only supported for 3 tasks.")
            print("Generating OOF CTR predictions for adaptive_hybrid strategy...")
            self.oof_ctr_preds_ = self._generate_oof_ctr_predictions(X, y_multi)
            print("OOF CTR predictions generated.")

        self.initial_predictions_ = self._compute_initial_predictions(y_multi)
        current_predictions = np.tile(self.initial_predictions_, (X.shape[0], 1))
        
        start_time = time.time()
        
        for i in range(self.n_estimators):
            raw_gradients, raw_hessians = self._compute_raw_gradients_hessians(y_multi, current_predictions)
            split_gradients, split_hessians = self._prepare_gradients_for_splitting(raw_gradients, raw_hessians, current_predictions, i)

            if self.subsample < 1.0:
                sample_indices = np.random.choice(X.shape[0], size=int(X.shape[0] * self.subsample), replace=False)
            else:
                sample_indices = np.arange(X.shape[0])

            X_sampled = X[sample_indices]
            
            feature_indices = None
            if self.colsample_bytree < 1.0:
                n_features = X.shape[1]
                feature_indices = np.random.choice(n_features, size=int(n_features * self.colsample_bytree), replace=False)
                X_sampled = X_sampled[:, feature_indices]

            tree = MultiTaskDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_tasks=self.n_tasks,
                weighting_strategy=self.weighting_strategy,
                gain_threshold=self.gain_threshold,
                track_split_gains=self.track_split_gains,
                verbose_logging=self.verbose_logging,
                is_dynamic_weight=self.is_dynamic_weight,
                gamma=self.gamma,
                delta=self.delta,
                random_state=self.random_state,
                threshold_prop_ctcvr=self.threshold_prop_ctcvr,
                threshold_prop_cvr=self.threshold_prop_cvr
            )

            tree.fit(
                X_sampled, 
                raw_gradients[sample_indices], 
                raw_hessians[sample_indices],
                split_gradients[sample_indices],
                split_hessians[sample_indices],
                y_true=y_multi[sample_indices],
                current_predictions=current_predictions[sample_indices]
            )
            
            self.trees_.append({'tree': tree, 'feature_indices': feature_indices})
            
            update = tree.predict(X_sampled)
            np.add.at(current_predictions, sample_indices, self.learning_rate * update)

            if (i + 1) % 10 == 0 or i == 0 or i == self.n_estimators - 1:
                elapsed_time = time.time() - start_time
                loss_value = self._calculate_loss(y_multi, current_predictions)
                loss_name = "LogLoss" if self.loss == "logloss" else "MSE"
                print(f"Iteration {i+1}/{self.n_estimators}, {loss_name}: {loss_value:.6f}, Time: {elapsed_time:.2f}s")

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        学習済みモデルで予測
        """
        X, _ = self._validate_input(X)
        
        if not self.trees_:
            raise ValueError("Model has not been trained yet")
        
        y_pred = np.tile(self.initial_predictions_, (X.shape[0], 1))
        
        for tree_info in self.trees_:
            tree = tree_info['tree']
            feature_indices = tree_info['feature_indices']
            
            X_subset = X[:, feature_indices] if feature_indices is not None else X
            tree_prediction = tree.predict(X_subset)
            
            y_pred += self.learning_rate * tree_prediction
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        確率予測を行う
        """
        if self.weighting_strategy == 'ordinal':
            # 2次元logit→3タスク確率
            logits = np.tile(self.initial_predictions_, (X.shape[0], 1))
            for tree_info in self.trees_:
                tree = tree_info['tree']
                X_subset = X
                tree_prediction = tree.predict(X_subset)
                logits += self.learning_rate * tree_prediction
            p1 = 1 / (1 + np.exp(-logits[:, 0]))
            p2 = 1 / (1 + np.exp(-logits[:, 1]))
            ctr = p1
            ctcvr = p2
            cvr = np.zeros_like(ctr)
            mask = ctr > 1e-7
            cvr[mask] = ctcvr[mask] / ctr[mask]
            cvr[~mask] = 0.0
            return np.stack([ctr, ctcvr, cvr], axis=1)
        else:
            logits = self.predict(X)
            y_proba = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        
            if (hasattr(self, 'weighting_strategy') and 
                self.weighting_strategy == "mtgbm-ctr-cvr" and 
                self.n_tasks == 3):
                y_proba[:, 1] = y_proba[:, 0] * y_proba[:, 2]
            
            return y_proba
    
    def get_feature_importance(self) -> np.ndarray:
        """
        特徴量の重要度を計算
        """
        if not self.trees_:
            raise ValueError("Model has not been trained yet")
        
        n_features = max(max(tree_info['feature_indices']) for tree_info in self.trees_ if tree_info['feature_indices'] is not None) + 1
        feature_importance = np.zeros(n_features)
        
        for tree_info in self.trees_:
            tree = tree_info['tree']
            self._count_feature_usage(tree.root, feature_importance)
        
        if np.sum(feature_importance) > 0:
            feature_importance = feature_importance / np.sum(feature_importance)
        
        return feature_importance
    
    def _count_feature_usage(self, node: DecisionTreeNode, feature_importance: np.ndarray) -> None:
        """
        ノードで使用される特徴量をカウント
        """
        if node is None or node.is_leaf:
            return
        
        if node.feature_idx is not None:
            feature_importance[node.feature_idx] += 1
        
        self._count_feature_usage(node.left, feature_importance)
        self._count_feature_usage(node.right, feature_importance)
    
    def get_node_logs(self) -> List[Dict]:
        """
        全ての木のノードログを取得
        """
        all_logs = []
        for i, tree_info in enumerate(self.trees_):
            tree = tree_info['tree']
            for log in tree.node_logs:
                log_copy = log.copy()
                log_copy['tree_index'] = i
                all_logs.append(log_copy)
        return all_logs
    
    def print_node_summary(self, tree_index: Optional[int] = None):
        """
        ノードログの要約を出力
        """
        if tree_index is not None:
            if tree_index >= len(self.trees_):
                print(f"Tree index {tree_index} is out of range.")
                return
            logs = self.trees_[tree_index]['tree'].node_logs
            print(f"\n=== Tree {tree_index} Node Summary ===")
        else:
            logs = self.get_node_logs()
            print(f"\n=== All Trees Node Summary ({len(self.trees_)} trees) ===")
        
        if not logs:
            print("No node logs available.")
            return
        
        total_nodes = len(logs)
        leaf_nodes = sum(1 for log in logs if log['is_leaf'])
        print(f"Total nodes: {total_nodes}, Split nodes: {total_nodes - leaf_nodes}, Leaf nodes: {leaf_nodes}")

    def save_logs_to_json(self, file_path: str):
        """
        ノードログをJSONファイルに保存
        """
        all_logs = self.get_node_logs()
        for log in all_logs:
            log['timestamp'] = datetime.now().isoformat()
        with open(file_path, 'w') as json_file:
            json.dump(all_logs, json_file, ensure_ascii=False, indent=4)
        print(f"Node logs saved to {file_path}")

    def _validate_input(self, X: np.ndarray, y_multi: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        入力データの検証と前処理
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if y_multi is not None:
            if y_multi.ndim == 1:
                y_multi = y_multi.reshape(-1, 1)
            if X.shape[0] != y_multi.shape[0]:
                raise ValueError("X and y_multi have different numbers of samples")
            data_n_tasks = y_multi.shape[1]
            # ordinal戦略用: 2列でもOK
            if self.weighting_strategy == 'ordinal':
                if data_n_tasks != 2:
                    raise ValueError("ordinal戦略は2列(click, cv)データのみ対応です")
                self.n_tasks = 2
            elif data_n_tasks == 2 and self.n_tasks == 3:
                y_multi = _add_cvr_labels(y_multi)
                self.n_tasks = y_multi.shape[1]
            else:
                self.n_tasks = y_multi.shape[1]
        return X, y_multi