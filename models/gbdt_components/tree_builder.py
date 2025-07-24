"""
Tree Builder

This module handles the construction of decision trees for multi-task learning,
including split finding, leaf value computation, and tree building logic.
"""

import numpy as np
from typing import Tuple, Optional, List
from .tree_node import DecisionTreeNode


class TreeBuilder:
    """
    決定木構築を担当するクラス
    
    Attributes:
    -----------
    max_depth : int
        最大深度
    min_samples_split : int
        分割に必要な最小サンプル数
    min_samples_leaf : int
        リーフノードに必要な最小サンプル数
    n_tasks : int
        タスク数
    node_counter : int
        ノードカウンター
    """
    
    def __init__(
        self,
        max_depth: int = 6,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        n_tasks: int = 2
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_tasks = n_tasks
        self.node_counter = 0
        
    def build_tree(
        self,
        X: np.ndarray,
        raw_gradients: np.ndarray,
        raw_hessians: np.ndarray,
        ensemble_gradients: np.ndarray,
        ensemble_hessians: np.ndarray,
        task_correlations: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        current_predictions: Optional[np.ndarray] = None
    ) -> DecisionTreeNode:
        """
        決定木を構築
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        raw_gradients : array-like, shape=(n_samples, n_tasks)
            生の勾配
        raw_hessians : array-like, shape=(n_samples, n_tasks)
            生のヘシアン
        ensemble_gradients : array-like, shape=(n_samples, n_tasks)
            アンサンブル勾配
        ensemble_hessians : array-like, shape=(n_samples, n_tasks)
            アンサンブルヘシアン
        task_correlations : array-like, shape=(n_tasks, n_tasks)
            タスク間相関行列
        y_true : array-like, shape=(n_samples, n_tasks), optional
            真のターゲット値
        current_predictions : array-like, shape=(n_samples, n_tasks), optional
            現在の予測値
            
        Returns:
        --------
        root : DecisionTreeNode
            構築された決定木のルートノード
        """
        self.node_counter = 0
        root = DecisionTreeNode(node_id=self.node_counter, depth=0)
        
        self._build_tree_recursive(
            root, X, raw_gradients, raw_hessians, ensemble_gradients, ensemble_hessians,
            task_correlations, 0, y_true, current_predictions
        )
        
        return root
    
    def _build_tree_recursive(
        self,
        node: DecisionTreeNode,
        X: np.ndarray,
        raw_gradients: np.ndarray,
        raw_hessians: np.ndarray,
        ensemble_gradients: np.ndarray,
        ensemble_hessians: np.ndarray,
        task_correlations: np.ndarray,
        depth: int,
        y_true: Optional[np.ndarray] = None,
        current_predictions: Optional[np.ndarray] = None
    ) -> None:
        """
        再帰的に決定木を構築
        
        Parameters:
        -----------
        node : DecisionTreeNode
            現在のノード
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        raw_gradients : array-like, shape=(n_samples, n_tasks)
            生の勾配
        raw_hessians : array-like, shape=(n_samples, n_tasks)
            生のヘシアン
        ensemble_gradients : array-like, shape=(n_samples, n_tasks)
            アンサンブル勾配
        ensemble_hessians : array-like, shape=(n_samples, n_tasks)
            アンサンブルヘシアン
        task_correlations : array-like, shape=(n_tasks, n_tasks)
            タスク間相関行列
        depth : int
            現在の深さ
        y_true : array-like, shape=(n_samples, n_tasks), optional
            真のターゲット値
        current_predictions : array-like, shape=(n_samples, n_tasks), optional
            現在の予測値
        """
        n_samples = X.shape[0]
        
        # ノード情報の設定
        node.node_id = self.node_counter
        node.depth = depth
        node.n_samples = n_samples
        self.node_counter += 1
        
        # CTR率の計算
        if y_true is not None and n_samples > 0:
            if self.n_tasks == 1:
                node.ctr_rate = np.mean(y_true[:, 0])
            else:
                # マルチタスクの場合、タスク0をCTRタスクと仮定
                ctr_mask = ~np.isnan(y_true[:, 0])
                if np.any(ctr_mask):
                    node.ctr_rate = np.mean(y_true[ctr_mask, 0])
                else:
                    node.ctr_rate = 0.0
        
        # 終了条件のチェック
        if self._should_stop_splitting(n_samples, depth):
            # リーフノードとして設定
            node.is_leaf = True
            node.values = self._compute_leaf_values(raw_gradients, raw_hessians)
            return
        
        # 最適な分割を探索
        best_split = self._search_best_split(X, ensemble_gradients, ensemble_hessians)
        
        if best_split is None:
            # 分割が見つからない場合、リーフノードとして設定
            node.is_leaf = True
            node.values = self._compute_leaf_values(raw_gradients, raw_hessians)
            return
        
        # 分割情報を設定
        feature_idx, threshold, information_gain = best_split
        node.feature_idx = feature_idx
        node.threshold = threshold
        node.information_gain = information_gain
        
        # データを分割
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # 左の子ノードを作成
        if np.sum(left_mask) > 0:
            node.left = DecisionTreeNode(depth=depth + 1)
            self._build_tree_recursive(
                node.left,
                X[left_mask],
                raw_gradients[left_mask],
                raw_hessians[left_mask],
                ensemble_gradients[left_mask],
                ensemble_hessians[left_mask],
                task_correlations,
                depth + 1,
                y_true[left_mask] if y_true is not None else None,
                current_predictions[left_mask] if current_predictions is not None else None
            )
        
        # 右の子ノードを作成
        if np.sum(right_mask) > 0:
            node.right = DecisionTreeNode(depth=depth + 1)
            self._build_tree_recursive(
                node.right,
                X[right_mask],
                raw_gradients[right_mask],
                raw_hessians[right_mask],
                ensemble_gradients[right_mask],
                ensemble_hessians[right_mask],
                task_correlations,
                depth + 1,
                y_true[right_mask] if y_true is not None else None,
                current_predictions[right_mask] if current_predictions is not None else None
            )
    
    def _should_stop_splitting(self, n_samples: int, depth: int) -> bool:
        """
        分割を停止するかどうかを判定
        
        Parameters:
        -----------
        n_samples : int
            サンプル数
        depth : int
            現在の深さ
            
        Returns:
        --------
        should_stop : bool
            分割を停止するかどうか
        """
        return (
            depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_samples < 2 * self.min_samples_leaf
        )
    
    def _search_best_split(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray
    ) -> Optional[Tuple[int, float, float]]:
        """
        最適な分割を探索
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        gradients : array-like, shape=(n_samples, n_tasks)
            勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            ヘシアン
            
        Returns:
        --------
        best_split : tuple or None
            最適な分割 (feature_idx, threshold, information_gain)
        """
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        # 各特徴量について分割を試行
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # 分割候補の閾値を生成
            if len(unique_values) < 2:
                continue
            
            thresholds = []
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                thresholds.append(threshold)
            
            # 各閾値について情報利得を計算
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # 最小サンプル数チェック
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # 情報利得を計算
                gain = self._calculate_information_gain(
                    gradients, hessians, left_mask, right_mask
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_feature is not None:
            return best_feature, best_threshold, best_gain
        else:
            return None
    
    def _calculate_information_gain(
        self,
        gradients: np.ndarray,
        hessians: np.ndarray,
        left_mask: np.ndarray,
        right_mask: np.ndarray
    ) -> float:
        """
        情報利得を計算
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            ヘシアン
        left_mask : array-like, shape=(n_samples,)
            左の分割マスク
        right_mask : array-like, shape=(n_samples,)
            右の分割マスク
            
        Returns:
        --------
        gain : float
            情報利得
        """
        # 全体の損失
        total_loss = self._calculate_node_loss(gradients, hessians)
        
        # 左の子ノードの損失
        left_loss = self._calculate_node_loss(gradients[left_mask], hessians[left_mask])
        
        # 右の子ノードの損失
        right_loss = self._calculate_node_loss(gradients[right_mask], hessians[right_mask])
        
        # 重み付き平均損失
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)
        n_total = n_left + n_right
        
        if n_total == 0:
            return 0.0
        
        weighted_loss = (n_left / n_total) * left_loss + (n_right / n_total) * right_loss
        
        # 情報利得
        gain = total_loss - weighted_loss
        
        return gain
    
    def _calculate_node_loss(self, gradients: np.ndarray, hessians: np.ndarray) -> float:
        """
        ノードの損失を計算
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            ヘシアン
            
        Returns:
        --------
        loss : float
            ノードの損失
        """
        if gradients.shape[0] == 0:
            return 0.0
        
        # 各タスクの損失を計算
        total_loss = 0.0
        
        for task_idx in range(self.n_tasks):
            task_gradients = gradients[:, task_idx]
            task_hessians = hessians[:, task_idx]
            
            # 有効なデータのマスク
            valid_mask = ~np.isnan(task_gradients) & ~np.isnan(task_hessians)
            
            if np.sum(valid_mask) > 0:
                valid_gradients = task_gradients[valid_mask]
                valid_hessians = task_hessians[valid_mask]
                
                # 損失計算: -0.5 * sum(g^2 / (h + lambda))
                regularization = 1e-6
                task_loss = -0.5 * np.sum(
                    valid_gradients**2 / (valid_hessians + regularization)
                )
                total_loss += task_loss
        
        return total_loss
    
    def _compute_leaf_values(
        self,
        gradients: np.ndarray,
        hessians: np.ndarray
    ) -> np.ndarray:
        """
        リーフノードの値を計算
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            ヘシアン
            
        Returns:
        --------
        values : array-like, shape=(n_tasks,)
            リーフノードの値
        """
        values = np.zeros(self.n_tasks)
        
        for task_idx in range(self.n_tasks):
            task_gradients = gradients[:, task_idx]
            task_hessians = hessians[:, task_idx]
            
            # 有効なデータのマスク
            valid_mask = ~np.isnan(task_gradients) & ~np.isnan(task_hessians)
            
            if np.sum(valid_mask) > 0:
                valid_gradients = task_gradients[valid_mask]
                valid_hessians = task_hessians[valid_mask]
                
                # リーフ値計算: -sum(g) / (sum(h) + lambda)
                grad_sum = np.sum(valid_gradients)
                hessian_sum = np.sum(valid_hessians)
                regularization = 1e-6
                
                values[task_idx] = -grad_sum / (hessian_sum + regularization)
            else:
                values[task_idx] = 0.0
        
        return values
