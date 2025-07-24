"""
Multi-Task Random Forest (MTRF) Module

This module contains the MTRF class that implements a multi-task random forest
using bagging with MultiTaskDecisionTree as base learners.
"""

import numpy as np
from typing import Optional, Union, List, Dict, Tuple
from datetime import datetime
import json

from .multi_task_tree import MultiTaskDecisionTree
from .data_transforms import _add_cvr_labels


class MTRF:
    """
    Multi-Task Random Forest implementation
    
    A random forest implementation using MultiTaskDecisionTree as base learners
    with bagging for multi-task learning scenarios.
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
        Initialize MTRF
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees
        max_depth : int
            Maximum depth of each tree
        min_samples_split : int
            Minimum samples required to split a node
        min_samples_leaf : int
            Minimum samples required at a leaf node
        n_tasks : int, optional
            Number of tasks (inferred from data if None)
        max_features : str, int, float or None
            Number of features to consider for each tree
        bootstrap : bool
            Whether to use bootstrap sampling
        weighting_strategy : str
            Weighting strategy for multi-task learning
        gamma : float
            Gamma parameter for weighting
        delta : float
            Delta parameter for weighting
        loss : str
            Loss function ("mse" or "logloss")
        random_state : int, optional
            Random seed
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
        
        # Initialize storage
        self.trees = []
        self.initial_predictions = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_n_features(self, n_total_features: int) -> int:
        """
        Calculate number of features to use for each tree
        
        Parameters:
        -----------
        n_total_features : int
            Total number of features
            
        Returns:
        --------
        n_features : int
            Number of features to use
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
        Fit the MTRF model
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            Training features
        y : array-like, shape=(n_samples, n_tasks)
            Training targets
        **kwargs : dict
            Additional parameters
            
        Returns:
        --------
        self : MTRF
            Fitted model
        """
        # Validate input
        X, y = self._validate_input(X, y)
        
        n_samples, n_features = X.shape
        
        # Set n_tasks if not provided
        if self.n_tasks is None:
            self.n_tasks = y.shape[1]
        
        # Compute initial predictions (mean of targets)
        self.initial_predictions = np.mean(y, axis=0)
        
        # Train each tree
        for i in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
            else:
                indices = np.arange(n_samples)
            
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Feature subsampling
            n_features_to_use = self._get_n_features(n_features)
            feature_indices = np.random.choice(n_features, size=n_features_to_use, replace=False)
            X_subset = X_bootstrap[:, feature_indices]
            
            # Create and fit tree
            tree = MultiTaskDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_tasks=self.n_tasks,
                weighting_strategy=self.weighting_strategy,
                gamma=self.gamma,
                delta=self.delta,
                random_state=self.random_state
            )
            
            # For RF, gradients are the targets themselves and hessians are ones
            gradients = y_bootstrap
            hessians = np.ones_like(y_bootstrap)
            
            tree.fit(X_subset, gradients, hessians)
            
            # Store tree with feature indices
            self.trees.append({
                'tree': tree,
                'feature_indices': feature_indices
            })
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            Input features
            
        Returns:
        --------
        predictions : array-like, shape=(n_samples, n_tasks)
            Predictions for each task
        """
        if self.initial_predictions is None:
            raise ValueError("Model has not been fitted yet")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_tasks = len(self.initial_predictions)
        
        # Initialize predictions
        predictions = np.zeros((n_samples, n_tasks))
        
        # Average predictions from all trees
        for tree_info in self.trees:
            tree = tree_info['tree']
            feature_indices = tree_info['feature_indices']
            
            # Apply feature subsampling
            X_subset = X[:, feature_indices]
            tree_predictions = tree.predict(X_subset)
            
            # Ensure 2D shape
            if tree_predictions.ndim == 1:
                tree_predictions = tree_predictions.reshape(-1, 1)
            
            predictions += tree_predictions
        
        # Take average
        predictions /= self.n_estimators
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            Input features
            
        Returns:
        --------
        probabilities : array-like, shape=(n_samples, n_tasks)
            Predicted probabilities
        """
        # Get logits
        logits = self.predict(X)
        
        # Convert to probabilities
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        
        # Special handling for 3-task CTR-CVR strategy
        if (self.weighting_strategy == "mtgbm-ctr-cvr" and self.n_tasks == 3):
            # CTCVR = CTR × CVR
            probabilities[:, 1] = probabilities[:, 0] * probabilities[:, 2]
        
        return probabilities
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores
        
        Returns:
        --------
        feature_importance : array-like
            Feature importance scores
        """
        if not self.trees:
            raise ValueError("Model has not been fitted yet")
        
        # Get number of features from original data
        # We need to infer this from the maximum feature index used
        max_feature_idx = 0
        for tree_info in self.trees:
            feature_indices = tree_info['feature_indices']
            if len(feature_indices) > 0:
                max_feature_idx = max(max_feature_idx, np.max(feature_indices))
        
        n_features = max_feature_idx + 1
        feature_importance = np.zeros(n_features)
        
        # Accumulate importance from all trees
        for tree_info in self.trees:
            tree = tree_info['tree']
            feature_indices = tree_info['feature_indices']
            
            # Get tree importance
            tree_importance = tree.get_feature_importance()
            
            # Map back to original feature space
            for i, orig_idx in enumerate(feature_indices):
                if i < len(tree_importance):
                    feature_importance[orig_idx] += tree_importance[i]
        
        # Normalize
        if np.sum(feature_importance) > 0:
            feature_importance = feature_importance / np.sum(feature_importance)
        
        return feature_importance
    
    def get_node_logs(self) -> List[Dict]:
        """
        Get node logs from all trees
        
        Returns:
        --------
        all_logs : List[Dict]
            Combined node logs from all trees
        """
        all_logs = []
        
        for i, tree_info in enumerate(self.trees):
            tree = tree_info['tree']
            if hasattr(tree, 'node_logs'):
                for log in tree.node_logs:
                    log_copy = log.copy()
                    log_copy['tree_index'] = i
                    all_logs.append(log_copy)
        
        return all_logs
    
    def print_node_summary(self, tree_index: Optional[int] = None) -> None:
        """
        Print node summary statistics
        
        Parameters:
        -----------
        tree_index : int, optional
            Specific tree index (None for all trees)
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
        
        # Calculate statistics
        total_nodes = len(logs)
        leaf_nodes = sum(1 for log in logs if log['is_leaf'])
        split_nodes = total_nodes - leaf_nodes
        
        # Task-specific statistics (assuming CTR rate is available)
        if 'ctr_rate' in logs[0]:
            ctr_rates = [log['ctr_rate'] for log in logs]
            avg_ctr = np.mean(ctr_rates)
            min_ctr = np.min(ctr_rates)
            max_ctr = np.max(ctr_rates)
        else:
            avg_ctr = min_ctr = max_ctr = 0.0
        
        # Information gain statistics
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
        if 'ctr_rate' in logs[0]:
            print(f"CTR rate - Avg: {avg_ctr:.4f}, Min: {min_ctr:.4f}, Max: {max_ctr:.4f}")
        if gains:
            print(f"Information gain - Avg: {avg_gain:.4f}, Min: {min_gain:.4f}, Max: {max_gain:.4f}")
        
        # Depth statistics
        depth_stats = {}
        for log in logs:
            depth = log['depth']
            if depth not in depth_stats:
                depth_stats[depth] = {'count': 0, 'ctr_sum': 0, 'gain_sum': 0, 'gain_count': 0}
            depth_stats[depth]['count'] += 1
            if 'ctr_rate' in log:
                depth_stats[depth]['ctr_sum'] += log['ctr_rate']
            if not log['is_leaf'] and log['information_gain'] > 0:
                depth_stats[depth]['gain_sum'] += log['information_gain']
                depth_stats[depth]['gain_count'] += 1
        
        print("\nDepth-wise statistics:")
        for depth in sorted(depth_stats.keys()):
            stats = depth_stats[depth]
            avg_ctr = stats['ctr_sum'] / stats['count'] if 'ctr_rate' in logs[0] else 0.0
            avg_gain = stats['gain_sum'] / stats['gain_count'] if stats['gain_count'] > 0 else 0.0
            print(f"  Depth {depth}: {stats['count']} nodes, CTR={avg_ctr:.4f}, Gain={avg_gain:.4f}")
    
    def save_logs_to_json(self, file_path: str) -> None:
        """
        Save node logs to JSON file
        
        Parameters:
        -----------
        file_path : str
            Path to save the JSON file
        """
        all_logs = self.get_node_logs()
        
        # Add timestamp
        for log in all_logs:
            log['timestamp'] = datetime.now().isoformat()
        
        # Save to JSON
        with open(file_path, 'w') as f:
            json.dump(all_logs, f, ensure_ascii=False, indent=4)
        
        print(f"Node logs saved to {file_path}")
    
    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Validate and transform input data
        
        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like, optional
            Target values
            
        Returns:
        --------
        X : np.ndarray
            Validated features
        y : np.ndarray or None
            Validated targets
        """
        # Convert X to 2D array
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if y is not None:
            # Convert y to 2D array
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            
            # Check sample consistency
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X ({X.shape[0]} samples) and y ({y.shape[0]} samples) have different numbers of samples")
            
            # Handle 3-task conversion if needed
            data_n_tasks = y.shape[1]
            if data_n_tasks == 2 and self.n_tasks == 3:
                y = _add_cvr_labels(y)
                data_n_tasks = 3
            
            # Update n_tasks if needed
            if data_n_tasks != self.n_tasks:
                self.n_tasks = data_n_tasks
        
        return X, y
    
    def print_training_summary(self) -> None:
        """
        Print training summary
        """
        print(f"\n=== MTRF Training Summary ===")
        print(f"Strategy: {self.weighting_strategy}")
        print(f"Tasks: {self.n_tasks}")
        print(f"Trees: {len(self.trees)}")
        print(f"Max depth: {self.max_depth}")
        print(f"Max features: {self.max_features}")
        print(f"Bootstrap: {self.bootstrap}")
        
        # Print feature importance if available
        if self.trees:
            try:
                importance = self.get_feature_importance()
                print(f"Feature importance shape: {importance.shape}")
                print(f"Top 5 features: {np.argsort(importance)[-5:][::-1]}")
            except Exception as e:
                print(f"Could not compute feature importance: {e}")
