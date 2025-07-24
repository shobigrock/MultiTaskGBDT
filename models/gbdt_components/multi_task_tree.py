"""
Multi-Task Decision Tree

This module contains the MultiTaskDecisionTree class that orchestrates
the tree building process using the various components.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from .tree_node import DecisionTreeNode
from .tree_builder import TreeBuilder
from .gradient_computer import GradientComputer
from .weighting_strategies import WeightingStrategyManager
from .data_transforms import validate_input_data


class MultiTaskDecisionTree:
    """
    マルチタスク決定木クラス
    
    各コンポーネントを統合して決定木を構築する
    """
    
    def __init__(
        self,
        n_tasks: int = 2,
        max_depth: int = 6,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        weighting_strategy: str = "mtgbm",
        gamma: float = 50.0,
        delta: float = 0.5,
        loss: str = "logloss",
        gain_threshold: float = 0.1,
        is_dynamic_weight: bool = False,
        track_split_gains: bool = True,
        random_state: Optional[int] = None
    ):
        self.n_tasks = n_tasks
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.weighting_strategy = weighting_strategy
        self.gamma = gamma
        self.delta = delta
        self.loss = loss
        self.gain_threshold = gain_threshold
        self.is_dynamic_weight = is_dynamic_weight
        self.track_split_gains = track_split_gains
        self.random_state = random_state
        
        # コンポーネントの初期化
        self.tree_builder = TreeBuilder(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_tasks=n_tasks
        )
        
        self.gradient_computer = GradientComputer(
            loss=loss,
            n_tasks=n_tasks
        )
        
        self.weighting_manager = WeightingStrategyManager(
            weighting_strategy=weighting_strategy,
            n_tasks=n_tasks,
            gamma=gamma,
            delta=delta,
            gain_threshold=gain_threshold,
            is_dynamic_weight=is_dynamic_weight
        )
        
        # 木の状態
        self.root = None
        self.is_fitted = False
        self.feature_importance = None
        
    def fit(
        self,
        X: np.ndarray,
        y_multi: np.ndarray,
        current_predictions: Optional[np.ndarray] = None
    ) -> 'MultiTaskDecisionTree':
        """
        決定木を訓練
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y_multi : array-like, shape=(n_samples, n_tasks)
            ターゲット値
        current_predictions : array-like, shape=(n_samples, n_tasks), optional
            現在の予測値
            
        Returns:
        --------
        self : MultiTaskDecisionTree
            訓練済みの決定木
        """
        # 入力データの検証
        X, y_multi = validate_input_data(X, y_multi)
        
        # 現在の予測値が提供されていない場合、初期予測値を使用
        if current_predictions is None:
            initial_preds = self.gradient_computer.compute_initial_predictions(y_multi)
            current_predictions = np.tile(initial_preds, (X.shape[0], 1))
        
        # 勾配とヘシアンの計算
        raw_gradients, raw_hessians = self.gradient_computer.compute_gradients_hessians(
            y_multi, current_predictions
        )
        
        # タスク間相関の計算
        task_correlations = self.gradient_computer.compute_task_correlations(raw_gradients)
        
        # 重み付けされた勾配とヘシアンの計算
        ensemble_gradients, ensemble_hessians = self.weighting_manager.compute_ensemble_weights_gradients_hessians(
            raw_gradients, raw_hessians, task_correlations, current_predictions, y_multi
        )
        
        # 決定木の構築
        self.root = self.tree_builder.build_tree(
            X, raw_gradients, raw_hessians, ensemble_gradients, ensemble_hessians,
            task_correlations, y_multi, current_predictions
        )
        
        # 特徴量重要度の計算
        self.feature_importance = self._compute_feature_importance(X.shape[1])
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        予測を実行
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        predictions : array-like, shape=(n_samples, n_tasks)
            予測値
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X, _ = validate_input_data(X)
        
        if self.root is None:
            # 木が構築されていない場合、ゼロ予測
            return np.zeros((X.shape[0], self.n_tasks))
        
        return self.root.predict(X)
    
    def _compute_feature_importance(self, n_features: int) -> np.ndarray:
        """
        特徴量重要度を計算
        
        Parameters:
        -----------
        n_features : int
            特徴量数
            
        Returns:
        --------
        importance : array-like, shape=(n_features,)
            特徴量重要度
        """
        importance = np.zeros(n_features)
        
        if self.root is not None:
            self._count_feature_usage(self.root, importance)
        
        # 正規化
        total_importance = np.sum(importance)
        if total_importance > 0:
            importance = importance / total_importance
        
        return importance
    
    def _count_feature_usage(self, node: DecisionTreeNode, importance: np.ndarray) -> None:
        """
        特徴量の使用回数を再帰的にカウント
        
        Parameters:
        -----------
        node : DecisionTreeNode
            現在のノード
        importance : array-like, shape=(n_features,)
            特徴量重要度配列
        """
        if node.is_leaf:
            return
        
        # 現在のノードの特徴量重要度を加算
        if node.feature_idx is not None:
            importance[node.feature_idx] += node.information_gain
        
        # 子ノードを再帰的に処理
        if node.left is not None:
            self._count_feature_usage(node.left, importance)
        if node.right is not None:
            self._count_feature_usage(node.right, importance)
    
    def get_info(self) -> Dict[str, Any]:
        """
        決定木の情報を取得
        
        Returns:
        --------
        info : dict
            決定木の情報
        """
        info = {
            "n_tasks": self.n_tasks,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "weighting_strategy": self.weighting_strategy,
            "gamma": self.gamma,
            "delta": self.delta,
            "loss": self.loss,
            "is_fitted": self.is_fitted
        }
        
        if self.is_fitted and self.root is not None:
            info.update({
                "actual_depth": self.root.get_depth(),
                "n_nodes": self.root.count_nodes(),
                "feature_importance": self.feature_importance.tolist() if self.feature_importance is not None else None
            })
        
        return info
    
    def __str__(self) -> str:
        """
        決定木の文字列表現
        """
        if not self.is_fitted:
            return f"MultiTaskDecisionTree(not fitted, n_tasks={self.n_tasks})"
        
        info = self.get_info()
        return f"MultiTaskDecisionTree(n_tasks={self.n_tasks}, depth={info.get('actual_depth', 0)}, nodes={info.get('n_nodes', 0)})"
    
    def __repr__(self) -> str:
        return self.__str__()
