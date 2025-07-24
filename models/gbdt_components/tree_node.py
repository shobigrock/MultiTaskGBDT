"""
Decision Tree Node Implementation

This module contains the DecisionTreeNode class that represents
individual nodes in the multi-task decision tree.
"""

from typing import Optional
import numpy as np


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
        self.node_id = node_id
        self.depth = depth
        self.n_samples = 0
        self.ctr_rate = 0.0
        self.information_gain = 0.0
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        単一ノードでの予測
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        predictions : array-like, shape=(n_samples, n_tasks)
            予測値
        """
        if self.is_leaf:
            values = self.values if self.values is not None else [0.0]
            if isinstance(values, (int, float)):
                values = [values]
            return np.tile(values, (X.shape[0], 1))
        
        # 分割に従って左右の子に振り分け
        mask = X[:, self.feature_idx] <= self.threshold
        
        # Get number of tasks from leaf values
        n_tasks = len(self.values) if self.values is not None else 1
        predictions = np.zeros((X.shape[0], n_tasks))
        
        if self.left is not None and np.any(mask):
            left_pred = self.left.predict(X[mask])
            if left_pred.ndim == 1:
                left_pred = left_pred.reshape(-1, 1)
            
            # Ensure shape compatibility
            if left_pred.shape[1] != n_tasks:
                # Pad or truncate to match n_tasks
                if left_pred.shape[1] < n_tasks:
                    left_pred = np.hstack([left_pred, np.zeros((left_pred.shape[0], n_tasks - left_pred.shape[1]))])
                else:
                    left_pred = left_pred[:, :n_tasks]
            
            predictions[mask] = left_pred
            
        if self.right is not None and np.any(~mask):
            right_pred = self.right.predict(X[~mask])
            if right_pred.ndim == 1:
                right_pred = right_pred.reshape(-1, 1)
            
            # Ensure shape compatibility
            if right_pred.shape[1] != n_tasks:
                # Pad or truncate to match n_tasks
                if right_pred.shape[1] < n_tasks:
                    right_pred = np.hstack([right_pred, np.zeros((right_pred.shape[0], n_tasks - right_pred.shape[1]))])
                else:
                    right_pred = right_pred[:, :n_tasks]
            
            predictions[~mask] = right_pred
            
        return predictions
    
    def get_depth(self) -> int:
        """
        このノードを根とする部分木の深さを計算
        
        Returns:
        --------
        depth : int
            部分木の深さ
        """
        if self.is_leaf:
            return 0
        
        left_depth = self.left.get_depth() if self.left else 0
        right_depth = self.right.get_depth() if self.right else 0
        
        return 1 + max(left_depth, right_depth)
    
    def count_nodes(self) -> int:
        """
        このノードを根とする部分木のノード数を計算
        
        Returns:
        --------
        count : int
            ノード数
        """
        if self.is_leaf:
            return 1
        
        left_count = self.left.count_nodes() if self.left else 0
        right_count = self.right.count_nodes() if self.right else 0
        
        return 1 + left_count + right_count
    
    def __str__(self) -> str:
        """
        ノードの文字列表現
        """
        if self.is_leaf:
            return f"Leaf(id={self.node_id}, depth={self.depth}, samples={self.n_samples}, values={self.values})"
        else:
            return f"Node(id={self.node_id}, depth={self.depth}, samples={self.n_samples}, feature={self.feature_idx}, threshold={self.threshold:.4f})"
    
    def __repr__(self) -> str:
        return self.__str__()
