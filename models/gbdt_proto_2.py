import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import time
import json
from datetime import datetime
from .base import MTGBMBase


def _add_cvr_labels(y: np.ndarray) -> np.ndarray:
    """
    Convert [Click, CV] binary labels to [CTR, CTCVR, CVR] format for 3-task learning.
    
    Parameters:
    -----------
    y : array-like, shape=(n_samples, 2)
        Binary labels [Click, CV] where:
        - y[:, 0] = Click (0 or 1)
        - y[:, 1] = CV (0 or 1)
    
    Returns:
    --------
    y_converted : array-like, shape=(n_samples, 3)
        Converted labels [CTR, CTCVR, CVR] where:
        - y_converted[:, 0] = CTR = Click
        - y_converted[:, 1] = CTCVR = Click * CV
        - y_converted[:, 2] = CVR = CV if Click=1, else NaN
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
    ctr_rate : float
        このノードでのCTR率（タスク0の平均値）
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
        self.ctr_rate = 0.0
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
                 n_tasks: int = 2,
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
            勾配強調のハイパーパラメータ
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
        
        # 分割利得の記録
        self.split_gains = []
        self.current_iteration = 0
        
        # ノード情報のログ
        self.node_logs = []
        self.node_counter = 0
        
        # 勾配重み（デフォルトはNone、均等重みを使用）
        self.gradient_weights = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, 
            X: np.ndarray, 
            gradients: np.ndarray, 
            hessians: np.ndarray, 
            task_correlations: Optional[np.ndarray] = None,
            y_true: Optional[np.ndarray] = None,
            iteration: int = 0,
            current_predictions: Optional[np.ndarray] = None) -> 'MultiTaskDecisionTree':
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
        current_predictions : array-like, shape=(n_samples, n_tasks), optional
            現在のアンサンブル予測値（新戦略用）
            
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
        # 3タスクモードでCVRタスクにNaN値がある場合はマスクを作成
        mask = None
        if self.n_tasks == 3:
            # CVRタスク（タスク2）のNaN値を除外するマスクを作成
            mask = ~np.isnan(gradients)
        
        raw_gradients = self._compute_normalization_weights(gradients, target_mean=0.05, target_std=0.01, mask=mask)
        raw_hessians = self._compute_normalization_weights(hessians, target_mean=1.0, target_std=0.1, mask=mask)
        
        # 2. 重み戦略に応じたアンサンブル勾配・ヘシアンを作成（分割探索で使用）
        gamma_limited = min(self.gamma, 10.0)  # ガンマ値を制限して数値安定性を向上
        
        ensemble_gradients = raw_gradients.copy()
        
        # タスク数1の場合は重み戦略を適用せずそのまま使用
        if self.n_tasks == 1:
            # シングルタスクの場合は重み戦略を適用しない
            pass
        elif self.n_tasks == 2:
            # 2タスクの場合の重み戦略
            # 乱数を用いて強調タスクを決定（CTR=0, CTCVR=1）
            if self.weighting_strategy == "mtgbm":
                emphasis_task = np.random.choice(2)
                ensemble_gradients[:, emphasis_task] *= gamma_limited

            # CTRタスク（インデックス0）の勾配にgammaを乗算        
            elif self.weighting_strategy == "proposed":
                ensemble_gradients[:, 0] *= gamma_limited  # CTR強調

            # CTCVRタスク（インデックス1）の勾配にgammaを乗算
            elif self.weighting_strategy == "proposed_reverse":
                ensemble_gradients[:, 1] *= gamma_limited  # CTCVR強調
            
            # CTRとCTCVRの重みを0.5:0.5に設定（均等重み）
            elif self.weighting_strategy == "half":
                ensemble_gradients[:, 0] *= 0.5  # CTR
                ensemble_gradients[:, 1] *= 0.5  # CTCVR
        elif self.n_tasks == 3:
            # 3タスクの場合の重み戦略
            if self.weighting_strategy == "mtgbm-three":
                # 3タスクから1つをランダム選択（1本目はCVRを除外）
                if hasattr(self, 'current_iteration') and self.current_iteration == 0:
                    emphasis_task = np.random.choice(2)  # CTR, CTCVRのみ
                else:
                    emphasis_task = np.random.choice(3)  # 全タスク
                ensemble_gradients[:, emphasis_task] *= gamma_limited
            
            elif self.weighting_strategy == "mtgbm-three-afterIPS":
                # mtgbm-threeと同じ重み戦略（IPS処理は後で適用）
                if hasattr(self, 'current_iteration') and self.current_iteration == 0:
                    emphasis_task = np.random.choice(2)  # CTR, CTCVRのみ
                else:
                    emphasis_task = np.random.choice(3)  # 全タスク
                ensemble_gradients[:, emphasis_task] *= gamma_limited
            
            elif self.weighting_strategy == "mtgbm-ctr-cvr":
                # CTRとCVRのみから1つをランダム選択（CTCVRは除外）
                emphasis_task = np.random.choice([0, 2])  # CTR(0), CVR(2)のみ
                ensemble_gradients[:, emphasis_task] *= gamma_limited
            
            elif self.weighting_strategy == "proposed-ctcvr-3":
                # CTRを初期強調、情報利得低下時にCTCVRに切り替え（CVRは除外）
                ensemble_gradients[:, 0] *= gamma_limited  # CTR強調
            
            elif self.weighting_strategy == "proposed-cvr-3":
                # 1回目はCTR強調、2回目以降はCVR強調、情報利得低下時にCTRに切り替え
                if hasattr(self, 'current_iteration') and self.current_iteration == 0:
                    ensemble_gradients[:, 0] *= gamma_limited  # 1回目はCTR強調
                else:
                    ensemble_gradients[:, 2] *= gamma_limited  # 2回目以降はCVR強調
            
            # 従来の戦略も3タスクに対応
            # elif self.weighting_strategy == "mtgbm":
            #     emphasis_task = np.random.choice(3)
            #     ensemble_gradients[:, emphasis_task] *= gamma_limited
            
            elif self.weighting_strategy == "proposed":
                ensemble_gradients[:, 0] *= gamma_limited  # CTR強調
            
            elif self.weighting_strategy == "proposed_reverse":
                ensemble_gradients[:, 1] *= gamma_limited  # CTCVR強調
            
            elif self.weighting_strategy == "half":
                # 3タスク均等重み
                ensemble_gradients[:, 0] *= 1/3  # CTR
                ensemble_gradients[:, 1] *= 1/3  # CTCVR
                ensemble_gradients[:, 2] *= 1/3  # CVR
        
        # ルートノードを作成し、再帰的に木を構築
        # raw_gradients/hessians: 予測値計算用（素の値）
        # ensemble_gradients/hessians: 分割探索用（重み付け済み）
        self.root = DecisionTreeNode()
        self._build_tree(
            self.root, X, raw_gradients, raw_hessians, ensemble_gradients, raw_hessians,
            task_correlations, depth=0, y_true=y_true, current_predictions=current_predictions
        )
        
        return self
    
    def _compute_ensemble_weights_gradients_hessians(self, 
                                                    gradients: np.ndarray, 
                                                    hessians: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        アンサンブル勾配・ヘシアン計算（アルゴリズム2）
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配（タスク0: CTR, タスク1: CTCVR）
        hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのヘシアン
            
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
            # CVRタスク（タスク2）のNaN値を除外するマスクを作成
            mask = ~np.isnan(gradients)
        
        # 1. 勾配の正規化計算（平均0.05、標準偏差0.01）
        normalized_gradients = self._compute_normalization_weights(gradients, target_mean=0.05, target_std=0.01, mask=mask)
        
        # 2. ヘシアンの正規化計算（平均1.0、標準偏差0.1）
        normalized_hessians = self._compute_normalization_weights(hessians, target_mean=1.00, target_std=0.1, mask=mask)
        
        # 4. 勾配とヘシアンの加重和を計算
        # gradient_weightsがない場合は均等重み
        if self.gradient_weights is None:
            weights = np.ones(n_tasks) / n_tasks
        else:
            weights = self.gradient_weights
        
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
            アンサンブル勾配（IPS重みづけ後）
        ensemble_hessians : array-like, shape=(n_samples,)
            アンサンブルヘシアン（IPS重みづけ後）
        """
        n_samples, n_tasks = gradients.shape
        
        # 1. 3タスクモードでCVRタスクにNaN値がある場合はマスクを作成
        mask = None
        if gradients.shape[1] == 3:
            mask = ~np.isnan(gradients)
        
        # 2. 勾配・ヘシアンを正規化（IPS適用前）
        normalized_gradients = self._compute_normalization_weights(gradients, target_mean=0.05, target_std=0.01, mask=mask)
        normalized_hessians = self._compute_normalization_weights(hessians, target_mean=1.00, target_std=0.1, mask=mask)
        
        # 3. CVRタスクにIPS重みづけを適用（正規化後）
        if self.n_tasks == 3:
            # CTR予測値を取得（タスク0、確率に変換）
            ctr_pred = probs[:, 0]
            
            # IPS重みを計算
            ips_weights = _compute_ips_weight(ctr_pred)
            
            # CVRタスク（タスク2）の正規化済み勾配・ヘシアンにIPS重みを適用
            # Click=1のサンプルのみ有効（NaNでないサンプル）
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
            有効なデータを示すマスク配列（TrueまたはFalseで構成）
            NaN値がある場合（CVRタスクなど）に使用
            
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
                    # 標準偏差が0の場合は目標平均値への単純シフト（有効データのみ）
                    normalized_data[task_mask, task_idx] = target_mean
                    # 無効データは元の値を保持（NaNなど）
                    normalized_data[~task_mask, task_idx] = task_data[~task_mask]
                else:
                    # 線形変換: (data - current_mean) / current_std * target_std + target_mean
                    normalized_data[task_mask, task_idx] = (valid_data - current_mean) / current_std * target_std + target_mean
                    # 無効データは元の値を保持（NaNなど）
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
                   raw_gradients: np.ndarray, 
                   raw_hessians: np.ndarray,
                   ensemble_gradients: np.ndarray,
                   ensemble_hessians: np.ndarray,
                   task_correlations: np.ndarray,
                   depth: int,
                   y_true: Optional[np.ndarray] = None,
                   current_predictions: Optional[np.ndarray] = None) -> None:
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
        current_predictions : array-like, shape=(n_samples, n_tasks), optional
            現在のアンサンブル予測値（新戦略用）
        """
        n_samples = X.shape[0]
        
        # ノード情報の設定
        node.node_id = self.node_counter
        node.depth = depth
        node.n_samples = n_samples
        self.node_counter += 1
        
        # CTR率の計算（タスク0をCTRタスクと仮定、シングルタスクの場合は単一値の率）
        if y_true is not None and n_samples > 0:
            if self.n_tasks == 1:
                # シングルタスクの場合は単一の値の平均
                node.ctr_rate = np.mean(y_true[:, 0])
            else:
                # マルチタスクの場合はタスク0をCTRタスクと仮定
                node.ctr_rate = np.mean(y_true[:, 0])
        else:
            # y_trueがない場合は素の勾配から推定（目安として）
            if self.n_tasks >= 1:
                node.ctr_rate = max(0.0, min(1.0, -np.mean(raw_gradients[:, 0]) if n_samples > 0 else 0.0))
            else:
                node.ctr_rate = 0.0
        
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
        if (hasattr(self, 'weighting_strategy') and 
            self.weighting_strategy == "mtgbm-three-afterIPS"):
            # 新戦略：正規化後IPS重みづけを適用
            # 現在の予測値を使用（current_predictionsが利用可能な場合）
            if current_predictions is not None:
                # 予測値を確率に変換
                current_probs = 1.0 / (1.0 + np.exp(-current_predictions))
                current_ensemble_gradients, current_ensemble_hessians = self._compute_ensemble_weights_gradients_hessians_afterIPS(
                    raw_gradients, raw_hessians, current_probs
                )
            else:
                # 予測値がない場合は従来方法を使用
                current_ensemble_gradients, current_ensemble_hessians = self._compute_ensemble_weights_gradients_hessians(
                    raw_gradients, raw_hessians
                )
        else:
            # 従来戦略：従来のアンサンブル勾配・ヘシアン計算
            current_ensemble_gradients, current_ensemble_hessians = self._compute_ensemble_weights_gradients_hessians(
                raw_gradients, raw_hessians
            )
        

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
        
        # 動的重み調整（マルチタスクの場合のみ）
        if self.n_tasks >= 2:
            if self.weighting_strategy == "proposed":
                
                # 情報利得がdelta閾値を下回る場合、重みを切り替え
                if best_gain < self.delta:
                    # CTRタスクの重みを下げ、CTCVRタスクの重みを上げる
                    adjusted_ensemble_gradients = ensemble_gradients.copy()
                    adjusted_ensemble_gradients[:, 0] /= self.gamma  # CTR重み削減
                    adjusted_ensemble_gradients[:, 1] *= self.gamma  # CTCVR重み増加
                    
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
                        
            elif self.weighting_strategy == "proposed-ctcvr-3":
                
                # 情報利得がdelta閾値を下回る場合、重みを切り替え（3タスク版）
                if best_gain < self.delta:
                    # CTRタスクの重みを下げ、CTCVRタスクの重みを上げる（CVRは除外）
                    adjusted_ensemble_gradients = ensemble_gradients.copy()
                    adjusted_ensemble_gradients[:, 0] /= self.gamma  # CTR重み削減
                    adjusted_ensemble_gradients[:, 1] *= self.gamma  # CTCVR重み増加
                    # CVRタスク（インデックス2）は重み調整なし
                    
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
                        
            elif self.weighting_strategy == "proposed-cvr-3":
                
                # 情報利得がdelta閾値を下回る場合、重みを切り替え（CVR→CTR）
                if best_gain < self.delta:
                    # 1回目イテレーションでは既にCTR強調済みなので切り替えなし
                    if not (hasattr(self, 'current_iteration') and self.current_iteration == 0):
                        # CVRタスクの重みを下げ、CTRタスクの重みを上げる（CTCVRは除外）
                        adjusted_ensemble_gradients = ensemble_gradients.copy()
                        adjusted_ensemble_gradients[:, 2] /= self.gamma  # CVR重み削減
                        adjusted_ensemble_gradients[:, 0] *= self.gamma  # CTR重み増加
                        # CTCVRタスク（インデックス1）は重み調整なし
                        
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
                    # CTCVRタスクの重みを下げ、CTRタスクの重みを上げる
                    adjusted_ensemble_gradients = ensemble_gradients.copy()
                    adjusted_ensemble_gradients[:, 1] /= self.gamma  # CTCVR重み削減
                    adjusted_ensemble_gradients[:, 0] *= self.gamma  # CTR重み増加
                    
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
            y_true[best_left_indices] if y_true is not None else None,
            current_predictions[best_left_indices] if current_predictions is not None else None
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
            y_true[best_right_indices] if y_true is not None else None,
            current_predictions[best_right_indices] if current_predictions is not None else None
        )
    
    def _compute_leaf_values(self, 
                            raw_gradients: np.ndarray, 
                            raw_hessians: np.ndarray,
                            task_correlations: np.ndarray,
                            y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        リーフノードの値を計算（素の勾配・ヘシアンを使用、CVRタスクはClick=1データのみ）
        
        Parameters:
        -----------
        raw_gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの素の勾配（重み付けされていない）
        raw_hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの素のヘシアン（重み付けされていない）
        task_correlations : array-like, shape=(n_tasks, n_tasks)
            タスク間の相関行列
        y_true : array-like, shape=(n_samples, n_tasks), optional
            実際のラベル値（未使用、勾配から一貫して計算）
            
        Returns:
        --------
        leaf_values : array-like, shape=(n_tasks,)
            各タスクのリーフ値
        """
        n_samples = raw_gradients.shape[0]
        n_tasks = raw_gradients.shape[1]
        
        # L2正則化パラメータ（XGBoostのデフォルトに合わせて小さく設定）
        lambda_reg = 0.01
        
        # 勾配から一貫してリーフ値を計算
        leaf_values = np.zeros(n_tasks)
        
        for task_idx in range(n_tasks):
            if task_idx == 2 and n_tasks >= 3:  # CVRタスクの特殊処理
                # CVRタスクはClick=1データのみで計算（NaNでないデータ）
                cvr_valid_mask = ~np.isnan(raw_gradients[:, 2])
                
                if np.any(cvr_valid_mask):
                    sum_gradients = np.sum(raw_gradients[cvr_valid_mask, task_idx])
                    sum_hessians = np.sum(raw_hessians[cvr_valid_mask, task_idx])
                    
                    if sum_hessians > 0:
                        # XGBoostの標準的な葉値計算: w* = -G / (H + λ)
                        leaf_values[task_idx] = -sum_gradients / (sum_hessians + lambda_reg)
                    else:
                        leaf_values[task_idx] = 0.0
                else:
                    # 0除算対策：CVRデータがない場合
                    leaf_values[task_idx] = 0.0
            else:  # CTR, CTCVRタスクの標準処理
                sum_gradients = np.sum(raw_gradients[:, task_idx])
                sum_hessians = np.sum(raw_hessians[:, task_idx])
                
                if sum_hessians > 0:
                    # XGBoostの標準的な葉値計算: w* = -G / (H + λ)
                    leaf_values[task_idx] = -sum_gradients / (sum_hessians + lambda_reg)
                else:
                    leaf_values[task_idx] = 0.0
        
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
            'ctr_rate': node.ctr_rate,
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
        
        return self.root.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        確率予測を行う（ログオッズ空間の予測値をシグモイド変換で確率に変換）
        
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
        
        # シグモイド関数で確率に変換（数値安定性のためクリッピング）
        y_proba = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        
        # mtgbm-ctr-cvr戦略の場合、CTCVR = CTR × CVRで計算
        if (hasattr(self, 'weighting_strategy') and 
            self.weighting_strategy == "mtgbm-ctr-cvr" and 
            self.n_tasks == 3):
            # CTR(タスク0) × CVR(タスク2) = CTCVR(タスク1)
            y_proba[:, 1] = y_proba[:, 0] * y_proba[:, 2]
        
        return y_proba


class MTGBDT(MTGBMBase):
    """
    汎用マルチタスク勾配ブースティング決定木（MT-GBDT）のスクラッチ実装
    
    シングルタスク（n_tasks=1）とマルチタスク（n_tasks>=2）の両方に対応。
    シングルタスクの場合は従来のGBDTと同等の動作をし、
    マルチタスクの場合は各タスク間の相関を活用した学習を行う。
    
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
        n_tasks: int = 2,
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
        n_tasks : int, default=2
            タスク数
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
        
        # 新しいパラメータの検証
        if self.weighting_strategy not in ["mtgbm", "mtgbm-three", "mtgbm-three-afterIPS", "mtgbm-ctr-cvr", "proposed", "proposed_reverse", "proposed-ctcvr-3", "proposed-cvr-3", "half"]:
            raise ValueError(f"Unsupported weighting strategy: {self.weighting_strategy}. Use 'mtgbm', 'mtgbm-three', 'mtgbm-three-afterIPS', 'mtgbm-ctr-cvr', 'proposed', 'proposed_reverse', 'proposed-ctcvr-3', 'proposed-cvr-3', or 'half'.")
        
        # 3タスク専用戦略のタスク数チェック（実際のフィットタイミングでチェック）
        self._strategy_task_requirements = {
            "mtgbm-three": 3,
            "mtgbm-three-afterIPS": 3,
            "mtgbm-ctr-cvr": 3, 
            "proposed-ctcvr-3": 3,
            "proposed-cvr-3": 3
        }
        
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
        初期予測値を計算（ログオッズ空間で計算、3タスク対応でCVRのNaN値を適切に処理）
        
        Parameters:
        -----------
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクのターゲット値（CVRはClick=0でNaN）
            
        Returns:
        --------
        initial_predictions : array-like, shape=(n_tasks,)
            各タスクの初期予測値（ログオッズ空間）
        """
        if self.n_tasks == 3:
            # 3タスクモード：CVRタスクのNaN値を適切に処理
            initial_preds = np.zeros(self.n_tasks)
            
            # CTRをログオッズに変換
            ctr_prob = np.clip(np.mean(y_multi[:, 0]), 1e-7, 1-1e-7)
            initial_preds[0] = np.log(ctr_prob / (1 - ctr_prob))  # CTR（ログオッズ）
            
            # CTCVRをログオッズに変換
            ctcvr_prob = np.clip(np.mean(y_multi[:, 1]), 1e-7, 1-1e-7)
            initial_preds[1] = np.log(ctcvr_prob / (1 - ctcvr_prob))  # CTCVR（ログオッズ）
            
            # CVRはClick=1のサンプルのみで平均計算し、ログオッズに変換
            cvr_mask = ~np.isnan(y_multi[:, 2])
            if np.sum(cvr_mask) > 0:
                cvr_prob = np.clip(np.mean(y_multi[cvr_mask, 2]), 1e-7, 1-1e-7)
                initial_preds[2] = np.log(cvr_prob / (1 - cvr_prob))  # CVR（ログオッズ）
            else:
                initial_preds[2] = 0.0  # デフォルト値（ログオッズ空間の中央値）
                
            return initial_preds
        else:
            # 2タスクモード：従来通りの処理をログオッズ空間で
            initial_preds = np.zeros(self.n_tasks)
            for task in range(self.n_tasks):
                task_prob = np.clip(np.mean(y_multi[:, task]), 1e-7, 1-1e-7)
                initial_preds[task] = np.log(task_prob / (1 - task_prob))
            return initial_preds
    
    def _compute_gradients_hessians(self, y_multi: np.ndarray, y_pred_logits: np.ndarray, iteration: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        勾配とヘシアンを計算（3タスク対応：CTR, CTCVR, CVR）
        ログオッズ空間の予測値を確率に変換してから勾配計算
        
        Parameters:
        -----------
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクのターゲット値（CVRはClick=0でNaN）
        y_pred_logits : array-like, shape=(n_samples, n_tasks)
            現在の予測値（ログオッズ空間）
        iteration : int, default=0
            現在のイテレーション数（IPS重み計算用）
            
        Returns:
        --------
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクのヘシアン
        """
        if self.loss == "mse":
            # 二乗誤差損失の勾配とヘシアン
            # ログオッズを確率に変換
            probs = 1 / (1 + np.exp(-np.clip(y_pred_logits, -500, 500)))
            
            # 勾配: 2 * (予測値 - 真の値)
            # ヘシアン: 2
            gradients = 2 * (probs - y_multi)
            hessians = np.ones_like(gradients) * 2
            
        elif self.loss == "logloss":
            # クロスエントロピー損失の勾配とヘシアン（二値分類）
            # ログオッズを確率に変換
            probs = 1 / (1 + np.exp(-np.clip(y_pred_logits, -500, 500)))
            
            # 勾配: p - y (where p = predicted probability)
            gradients = probs - y_multi
            
            # ヘシアン: p * (1 - p)
            hessians = probs * (1 - probs)
            # 数値安定性のため最小値を設定
            hessians = np.maximum(hessians, 1e-7)
            
            # 3タスクモードでCVRタスク（task 2）の特別処理
            if self.n_tasks == 3:
                # CTR予測値を取得（タスク0、確率に変換）
                ctr_pred = probs[:, 0]
                
                # mtgbm-three戦略で初期イテレーションの場合、CVRタスクの勾配を0にする
                if (hasattr(self, 'weighting_strategy') and 
                    self.weighting_strategy == "mtgbm-three" and 
                    iteration == 0):
                    gradients[:, 2] = 0.0
                    hessians[:, 2] = 0.0
                # mtgbm-three-afterIPS戦略では、ここではIPS重みづけを行わない（後で正規化後に適用）
                elif (hasattr(self, 'weighting_strategy') and 
                      self.weighting_strategy == "mtgbm-three-afterIPS"):
                    # NaNサンプルの勾配・ヘシアンを0にセットするのみ
                    nan_mask = np.isnan(y_multi[:, 2])
                    gradients[nan_mask, 2] = 0.0
                    hessians[nan_mask, 2] = 0.0
                else:
                    # IPS重みを計算（mtgbm-three、proposed-ctcvr-3等の3タスク戦略で適用）
                    ips_weights = _compute_ips_weight(ctr_pred)
                    
                    # CVRタスク（タスク2）の勾配・ヘシアンにIPS重みを適用
                    # Click=1のサンプルのみ有効（NaNでないサンプル）
                    cvr_mask = ~np.isnan(y_multi[:, 2])
                    gradients[cvr_mask, 2] *= ips_weights[cvr_mask]
                    hessians[cvr_mask, 2] *= ips_weights[cvr_mask]
                    
                    # NaNサンプルの勾配・ヘシアンを0にセット
                    nan_mask = np.isnan(y_multi[:, 2])
                    gradients[nan_mask, 2] = 0.0
                    hessians[nan_mask, 2] = 0.0
            
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")
        
        return gradients, hessians
    
    def _compute_task_correlations(self, gradients: np.ndarray) -> np.ndarray:
        """
        タスク間の相関行列を計算（3タスク対応でNaN値を適切に処理）
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの勾配
            
        Returns:
        --------
        task_correlations : array-like, shape=(n_tasks, n_tasks)
            タスク間の相関行列
        """
        # 3タスクモードでCVRタスクにNaN値がある場合の処理
        if self.n_tasks == 3:
            # 有効なデータのみで相関を計算
            valid_mask = ~(np.isnan(gradients).any(axis=1))
            if np.sum(valid_mask) > 1:
                gradients_valid = gradients[valid_mask]
                corr = np.corrcoef(gradients_valid.T)
            else:
                # 有効データが少ない場合は単位行列を返す
                corr = np.eye(self.n_tasks)
        else:
            # 従来の2タスクモードでは全データで計算
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
        シングル/マルチタスクデータでモデルを学習
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y_multi : array-like, shape=(n_samples, n_tasks)
            ターゲット値（n_tasks=1でシングルタスク、n_tasks>=2でマルチタスク）
        **kwargs : dict
            追加のパラメータ
            
        Returns:
        --------
        self : MTGBDT
            学習済みモデル
        """
        # 入力検証（3タスクデータ変換を含む）
        X, y_multi = self._validate_input(X, y_multi)
        
        # タスク数を取得
        data_n_tasks = y_multi.shape[1]
        
        # 戦略固有のタスク数チェック
        if hasattr(self, '_strategy_task_requirements'):
            required_tasks = self._strategy_task_requirements.get(self.weighting_strategy)
            if required_tasks and data_n_tasks != required_tasks:
                raise ValueError(f"Strategy '{self.weighting_strategy}' requires exactly {required_tasks} tasks, but got {data_n_tasks} tasks.")
            
        # 初期予測値を計算
        self.initial_predictions = self._compute_initial_predictions(y_multi)
        
        # 現在の予測値を初期化
        current_predictions = np.tile(self.initial_predictions, (X.shape[0], 1))
        
        # 学習開始時間
        start_time = time.time()
        
        # 各反復で木を構築
        for i in range(self.n_estimators):
            # 現在のイテレーション記録（mtgbm-three戦略用）
            self.current_iteration = i
            
            # 勾配とヘシアンを計算（3タスク対応、イテレーション情報を渡す）
            gradients, hessians = self._compute_gradients_hessians(y_multi, current_predictions, iteration=i)
            
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
                n_tasks=self.n_tasks,
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
                y_true=y_multi_subset,
                current_predictions=current_predictions
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
            current_predictions += self.learning_rate * update
            
            # 進捗を表示（オプション）
            if (i + 1) % 10 == 0 or i == 0 or i == self.n_estimators - 1:
                elapsed_time = time.time() - start_time
                
                if self.loss == "mse":
                    loss_value = np.mean((y_multi - current_predictions) ** 2)
                    loss_name = "MSE"
                elif self.loss == "logloss":
                    # logloss計算：ログオッズを確率に変換してから計算（3タスクのNaN値対応）
                    y_pred_proba = 1 / (1 + np.exp(-np.clip(current_predictions, -500, 500)))
                    y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
                    
                    # 3タスクの場合、CVRタスクのNaN値を無視してlogloss計算
                    if self.n_tasks == 3:
                        total_loss = 0.0
                        valid_samples = 0
                        
                        # CTRタスク（タスク0）
                        ctr_loss = -np.mean(y_multi[:, 0] * np.log(y_pred_proba[:, 0]) + 
                                           (1 - y_multi[:, 0]) * np.log(1 - y_pred_proba[:, 0]))
                        total_loss += ctr_loss
                        valid_samples += 1
                        
                        # CTCVRタスク（タスク1）
                        # mtgbm-ctr-cvr戦略の場合はCTCVRログ損失を除外
                        if not (hasattr(self, 'weighting_strategy') and 
                               self.weighting_strategy == "mtgbm-ctr-cvr"):
                            ctcvr_loss = -np.mean(y_multi[:, 1] * np.log(y_pred_proba[:, 1]) + 
                                                 (1 - y_multi[:, 1]) * np.log(1 - y_pred_proba[:, 1]))
                            total_loss += ctcvr_loss
                            valid_samples += 1
                        
                        # CVRタスク（タスク2）：NaN値を除外
                        cvr_valid_mask = ~np.isnan(y_multi[:, 2])
                        if np.sum(cvr_valid_mask) > 0:
                            cvr_loss = -np.mean(y_multi[cvr_valid_mask, 2] * np.log(y_pred_proba[cvr_valid_mask, 2]) + 
                                              (1 - y_multi[cvr_valid_mask, 2]) * np.log(1 - y_pred_proba[cvr_valid_mask, 2]))
                            total_loss += cvr_loss
                            valid_samples += 1
                        
                        loss_value = total_loss / valid_samples if valid_samples > 0 else float('inf')
                    else:
                        # 2タスクの場合：従来通り
                        loss_value = -np.mean(y_multi * np.log(y_pred_proba) + (1 - y_multi) * np.log(1 - y_pred_proba))
                    
                    loss_name = "LogLoss"
                else:
                    loss_value = np.mean((y_multi - current_predictions) ** 2)
                    loss_name = "Loss"
                
                print(f"Iteration {i+1}/{self.n_estimators}, {loss_name}: {loss_value:.6f}, Time: {elapsed_time:.2f}s")
                
                # デバッグ情報：3タスクでloglossがNaNの場合のみ
                if self.n_tasks == 3 and self.loss == "logloss" and np.isnan(loss_value):
                    print(f"  🚨 NaN detected! Debug info:")
                    cvr_valid = ~np.isnan(y_multi[:, 2])
                    print(f"    CVR valid samples: {np.sum(cvr_valid)}/{len(y_multi[:, 2])}")
                    if np.sum(cvr_valid) > 0:
                        y_pred_proba = 1 / (1 + np.exp(-np.clip(current_predictions, -500, 500)))
                        y_pred_proba = np.clip(y_pred_proba, 1e-7, 1 - 1e-7)
                        print(f"    CVR pred range: [{np.min(y_pred_proba[cvr_valid, 2]):.6f}, {np.max(y_pred_proba[cvr_valid, 2]):.6f}]")
                        print(f"    CVR true range: [{np.min(y_multi[cvr_valid, 2]):.6f}, {np.max(y_multi[cvr_valid, 2]):.6f}]")
        
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
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        確率予測を行う（ログオッズ空間の予測値をシグモイド変換で確率に変換）
        
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
        
        # シグモイド関数で確率に変換（数値安定性のためクリッピング）
        y_proba = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        
        # mtgbm-ctr-cvr戦略の場合、CTCVR = CTR × CVRで計算
        if (hasattr(self, 'weighting_strategy') and 
            self.weighting_strategy == "mtgbm-ctr-cvr" and 
            self.n_tasks == 3):
            # CTR(タスク0) × CVR(タスク2) = CTCVR(タスク1)
            y_proba[:, 1] = y_proba[:, 0] * y_proba[:, 2]
        
        return y_proba
    
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
        ctr_rates = [log['ctr_rate'] for log in logs]
        avg_cvr = np.mean(ctr_rates)
        min_cvr = np.min(ctr_rates)
        max_cvr = np.max(ctr_rates)
        
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
            depth_stats[depth]['cvr_sum'] += log['ctr_rate']
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

    def _validate_input(self, X: np.ndarray, y_multi: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        入力データの検証と前処理（3タスクモード対応）
        
        Parameters:
        -----------
        X : array-like
            入力特徴量
        y_multi : array-like, optional
            複数タスクのターゲット値
            
        Returns:
        --------
        X : np.ndarray
            検証・変換後の入力特徴量
        y_multi : np.ndarray or None
            検証・変換後のターゲット値（3タスク変換後）
        """
        # Xを2次元配列に変換
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # y_multiが指定されている場合
        if y_multi is not None:
            # y_multiを2次元配列に変換
            if y_multi.ndim == 1:
                y_multi = y_multi.reshape(-1, 1)
                
            # サンプル数の一致を確認
            if X.shape[0] != y_multi.shape[0]:
                raise ValueError(f"X ({X.shape[0]} samples) and y_multi ({y_multi.shape[0]} samples) have different numbers of samples")
            
            # 3タスクモードでの自動データ変換
            data_n_tasks = y_multi.shape[1]
            if data_n_tasks == 2 and self.n_tasks == 3:
                y_multi = _add_cvr_labels(y_multi)
                data_n_tasks = 3
            
            # タスク数を設定/検証
            if data_n_tasks != self.n_tasks:
                # データのタスク数で更新
                self.n_tasks = data_n_tasks
                    
        return X, y_multi


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
    n_tasks : int
        タスク数
    max_features : str, int, float or None
        各木で使用する特徴数の指定
    bootstrap : bool
        ブートストラップサンプリングを行うかどうか
    weighting_strategy : str
        重み戦略（"mtgbm", "proposed", "half"など）
    gamma : float
        勾配強調のハイパーパラメータ（proposed戦略用）
    delta : float
        勾配ベース重み調整のハイパーパラメータ（proposed戦略用）
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
                 max_depth: int = 6,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 n_tasks: Optional[int] = None,
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
        max_depth : int, default=6
            各木の最大深さ
        min_samples_split : int, default=2
            分割に必要な最小サンプル数
        min_samples_leaf : int, default=1
            リーフノードに必要な最小サンプル数
        n_tasks : int, optional
            タスク数。Noneの場合はfitメソッド内でデータから自動推定
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
            勾配ベース重み調整のハイパーパラメータ（proposed戦略用）
        loss : str, default="logloss"
            損失関数の種類（"mse", "logloss"）
        random_state : int, optional
            乱数シード
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_tasks = n_tasks
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.weighting_strategy = weighting_strategy
        self.gamma = gamma
        self.delta = delta
        self.loss = loss
        self.random_state = random_state
        
        # 学習済みの木を保存するリスト
        self.trees = []
        self.initial_predictions = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_n_features(self, n_total_features: int) -> int:
        """
        使用する特徴数を計算
        
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
            return int(np.log2(n_total_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_total_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_total_features))
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'MTRF':
        """
        マルチタスクランダムフォレストを学習
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y : array-like, shape=(n_samples, n_tasks) or (n_samples,)
            ターゲット値
        **kwargs : dict
            追加のパラメータ
            
        Returns:
        --------
        self : MTRF
            学習済みモデル
        """
        # 入力検証と変換
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        
        # 単一タスクの場合は2次元に変換
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples")
        
        n_samples, n_features = X.shape
        
        # n_tasksの設定: 初期化時に指定されていれば使用、そうでなければデータから推定
        if self.n_tasks is not None:
            n_tasks = self.n_tasks
        else:
            n_tasks = y.shape[1]
            # 自動推定された値を保存
            self.n_tasks = n_tasks
        
        # 初期予測値を計算（各タスクの平均値）
        self.initial_predictions = np.mean(y, axis=0)
        
        # 各木を学習
        for i in range(self.n_estimators):
            # ブートストラップサンプリング
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
            else:
                indices = np.arange(n_samples)
            
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # 特徴量サブサンプリング
            n_features_to_use = self._get_n_features(n_features)
            feature_indices = np.random.choice(n_features, size=n_features_to_use, replace=False)
            X_subset = X_bootstrap[:, feature_indices]
            
            # 新しい木を構築
            tree = MultiTaskDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_tasks=n_tasks,
                weighting_strategy=self.weighting_strategy,
                gamma=self.gamma,
                delta=self.delta,
                random_state=self.random_state
            )
            
            # 木を学習（MTRFでは勾配・ヘシアンを直接計算）
            # MSE損失の場合: gradient = y - y_pred, hessian = 1
            # 初期予測は0として、勾配は目標値そのもの
            gradients = y_bootstrap
            hessians = np.ones_like(y_bootstrap)
            
            tree.fit(X_subset, gradients, hessians)
            
            # 木と特徴量インデックスを保存
            self.trees.append({
                'tree': tree,
                'feature_indices': feature_indices
            })
        
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
        predictions : array-like, shape=(n_samples, n_tasks)
            各サンプル・各タスクの予測値
        """
        if self.initial_predictions is None:
            raise ValueError("Model has not been trained yet")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_tasks = len(self.initial_predictions)
        
        # 各木の予測値を平均
        predictions = np.zeros((n_samples, n_tasks))
        
        for tree_info in self.trees:
            tree = tree_info['tree']
            feature_indices = tree_info['feature_indices']
            
            # 特徴量サブセットで予測
            X_subset = X[:, feature_indices]
            tree_predictions = tree.predict(X_subset)
            
            # 単一タスクの場合は2次元に変換
            if tree_predictions.ndim == 1:
                tree_predictions = tree_predictions.reshape(-1, 1)
            
            predictions += tree_predictions
        
        # 平均を取る
        predictions /= self.n_estimators
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        確率予測を行う（ログオッズ空間の予測値をシグモイド変換で確率に変換）
        
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
        
        # シグモイド関数で確率に変換（数値安定性のためクリッピング）
        y_proba = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        
        # mtgbm-ctr-cvr戦略の場合、CTCVR = CTR × CVRで計算
        if (hasattr(self, 'weighting_strategy') and 
            self.weighting_strategy == "mtgbm-ctr-cvr" and 
            self.n_tasks == 3):
            # CTR(タスク0) × CVR(タスク2) = CTCVR(タスク1)
            y_proba[:, 1] = y_proba[:, 0] * y_proba[:, 2]
        
        return y_proba
    
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
        ctr_rates = [log['ctr_rate'] for log in logs]
        avg_cvr = np.mean(ctr_rates)
        min_cvr = np.min(ctr_rates)
        max_cvr = np.max(ctr_rates)
        
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
            depth_stats[depth]['cvr_sum'] += log['ctr_rate']
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

    def _validate_input(self, X: np.ndarray, y_multi: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        入力データの検証と前処理（3タスクモード対応）
        
        Parameters:
        -----------
        X : array-like
            入力特徴量
        y_multi : array-like, optional
            複数タスクのターゲット値
            
        Returns:
        --------
        X : np.ndarray
            検証・変換後の入力特徴量
        y_multi : np.ndarray or None
            検証・変換後のターゲット値（3タスク変換後）
        """
        # Xを2次元配列に変換
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # y_multiが指定されている場合
        if y_multi is not None:
            # y_multiを2次元配列に変換
            if y_multi.ndim == 1:
                y_multi = y_multi.reshape(-1, 1)
                
            # サンプル数の一致を確認
            if X.shape[0] != y_multi.shape[0]:
                raise ValueError(f"X ({X.shape[0]} samples) and y_multi ({y_multi.shape[0]} samples) have different numbers of samples")
            
            # 3タスクモードでの自動データ変換
            data_n_tasks = y_multi.shape[1]
            if data_n_tasks == 2 and self.n_tasks == 3:
                y_multi = _add_cvr_labels(y_multi)
                data_n_tasks = 3
            
            # タスク数を設定/検証
            if data_n_tasks != self.n_tasks:
                # データのタスク数で更新
                self.n_tasks = data_n_tasks
                    
        return X, y_multi