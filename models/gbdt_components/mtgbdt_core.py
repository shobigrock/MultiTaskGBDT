"""
MTGBDT Core Module

This module contains the main MTGBDT class that orchestrates all components
to provide a complete multi-task gradient boosting implementation.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
import json
import warnings

from .gradient_computer import GradientComputer
from .multi_task_tree import MultiTaskDecisionTree
from .data_transforms import _add_cvr_labels, _validate_strategy_n_tasks
from .weighting_strategies import WeightingStrategyManager


class MTGBMBase:
    """
    Base class for multi-task gradient boosting models
    """
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, 
                 max_depth: int = 3, random_state: Optional[int] = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)


class MTGBDT(MTGBMBase):
    """
    Multi-Task Gradient Boosting Decision Trees implementation
    
    A comprehensive implementation supporting both single-task (n_tasks=1) and
    multi-task (n_tasks>=2) scenarios with various weighting strategies.
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
                 random_state: Optional[int] = None):
        """
        Initialize MTGBDT
        
        Parameters:
        -----------
        n_estimators : int
            Number of boosting iterations
        learning_rate : float
            Learning rate for boosting
        max_depth : int
            Maximum depth of each tree
        min_samples_split : int
            Minimum samples required to split a node
        min_samples_leaf : int
            Minimum samples required at a leaf node
        n_tasks : int
            Number of tasks
        subsample : float
            Subsample ratio for training
        colsample_bytree : float
            Column subsample ratio for training
        gradient_weights : array-like, optional
            Weights for gradient computation
        normalize_gradients : bool
            Whether to normalize gradients
        loss : str
            Loss function ("mse" or "logloss")
        weighting_strategy : str
            Weighting strategy for multi-task learning
        gain_threshold : float
            Information gain threshold
        track_split_gains : bool
            Whether to track split gains
        is_dynamic_weight : bool
            Whether to use dynamic weighting
        gamma : float
            Gamma parameter for weighting
        delta : float
            Delta parameter for weighting
        random_state : int, optional
            Random seed
        """
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        
        # Store parameters
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
        
        # Initialize components
        self.gradient_computer = GradientComputer(
            n_tasks=n_tasks,
            loss=loss,
            normalize_gradients=normalize_gradients,
            weighting_strategy=weighting_strategy
        )
        
        self.weighting_manager = WeightingStrategyManager(
            weighting_strategy=weighting_strategy,
            gamma=gamma,
            delta=delta,
            gain_threshold=gain_threshold,
            is_dynamic_weight=is_dynamic_weight
        )
        
        # Validate strategy and task requirements
        _validate_strategy_n_tasks(weighting_strategy, n_tasks)
        
        # Initialize storage
        self.trees = []
        self.initial_predictions = None
        
        # Initialize tracking
        if self.track_split_gains:
            self.split_gains_history = []
            self.iteration_gains = []
            self.weight_switches = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'MTGBDT':
        """
        Fit the MTGBDT model
        
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
        self : MTGBDT
            Fitted model
        """
        # Validate and transform input
        X, y_multi = self._validate_input(X, y)
        
        # Compute initial predictions
        self.initial_predictions = self.gradient_computer.compute_initial_predictions(y_multi)
        
        # Initialize predictions with initial values
        n_samples = X.shape[0]
        current_predictions = np.tile(self.initial_predictions, (n_samples, 1))
        
        # Main boosting loop
        for iteration in range(self.n_estimators):
            # Compute gradients and hessians
            gradients, hessians = self.gradient_computer.compute_gradients_hessians(
                y_multi, current_predictions, iteration
            )
            
            # Handle subsample if needed
            if self.subsample < 1.0:
                sample_indices = np.random.choice(
                    n_samples, 
                    size=int(n_samples * self.subsample), 
                    replace=False
                )
                X_subset = X[sample_indices]
                gradients_subset = gradients[sample_indices]
                hessians_subset = hessians[sample_indices]
            else:
                X_subset = X
                gradients_subset = gradients
                hessians_subset = hessians
                sample_indices = None
            
            # Handle column subsample if needed
            if self.colsample_bytree < 1.0:
                n_features = X_subset.shape[1]
                feature_indices = np.random.choice(
                    n_features,
                    size=int(n_features * self.colsample_bytree),
                    replace=False
                )
                X_subset = X_subset[:, feature_indices]
            else:
                feature_indices = None
            
            # Create and fit tree
            tree = MultiTaskDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_tasks=self.n_tasks,
                weighting_strategy=self.weighting_strategy,
                gamma=self.gamma,
                delta=self.delta,
                gain_threshold=self.gain_threshold,
                is_dynamic_weight=self.is_dynamic_weight,
                track_split_gains=self.track_split_gains,
                random_state=self.random_state
            )
            
            tree.fit(X_subset, gradients_subset, hessians_subset)
            
            # Store tree with metadata
            tree_info = {
                'tree': tree,
                'sample_indices': sample_indices,
                'feature_indices': feature_indices
            }
            self.trees.append(tree_info)
            
            # Update predictions
            tree_predictions = self._predict_with_tree(X, tree_info)
            current_predictions += self.learning_rate * tree_predictions
            
            # Track split gains if enabled
            if self.track_split_gains and hasattr(tree, 'split_gains_history'):
                self.split_gains_history.extend(tree.split_gains_history)
                if tree.split_gains_history:
                    self.iteration_gains.append(np.mean(tree.split_gains_history))
                else:
                    self.iteration_gains.append(0.0)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the fitted model
        
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
        
        # Start with initial predictions
        predictions = np.tile(self.initial_predictions, (n_samples, 1))
        
        # Add predictions from each tree
        for tree_info in self.trees:
            tree_predictions = self._predict_with_tree(X, tree_info)
            predictions += self.learning_rate * tree_predictions
        
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
        # Get logits predictions
        logits = self.predict(X)
        
        # Convert to probabilities using sigmoid
        probabilities = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        
        # Special handling for 3-task strategies
        if (self.weighting_strategy == "mtgbm-ctr-cvr" and self.n_tasks == 3):
            # CTCVR = CTR × CVR
            probabilities[:, 1] = probabilities[:, 0] * probabilities[:, 2]
        
        return probabilities
    
    def _predict_with_tree(self, X: np.ndarray, tree_info: dict) -> np.ndarray:
        """
        Make predictions with a single tree considering subsampling
        
        Parameters:
        -----------
        X : array-like
            Input features
        tree_info : dict
            Tree information including subsampling indices
            
        Returns:
        --------
        predictions : array-like
            Tree predictions
        """
        tree = tree_info['tree']
        feature_indices = tree_info.get('feature_indices')
        
        # Apply feature subsampling if used
        if feature_indices is not None:
            X_subset = X[:, feature_indices]
        else:
            X_subset = X
        
        # Get predictions from tree
        predictions = tree.predict(X_subset)
        
        # Ensure 2D shape
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        return predictions
    
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
                # Update component configurations
                self.gradient_computer.n_tasks = data_n_tasks
        
        return X, y
    
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
        
        # Get number of features from first tree
        first_tree = self.trees[0]['tree']
        n_features = first_tree.get_n_features()
        
        # Initialize importance array
        feature_importance = np.zeros(n_features)
        
        # Accumulate importance from all trees
        for tree_info in self.trees:
            tree = tree_info['tree']
            tree_importance = tree.get_feature_importance()
            
            # Handle feature subsampling
            feature_indices = tree_info.get('feature_indices')
            if feature_indices is not None:
                # Map back to original feature space
                for i, orig_idx in enumerate(feature_indices):
                    if i < len(tree_importance):
                        feature_importance[orig_idx] += tree_importance[i]
            else:
                feature_importance += tree_importance
        
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
    
    def print_training_summary(self) -> None:
        """
        Print training summary
        """
        print(f"\n=== MTGBDT Training Summary ===")
        print(f"Strategy: {self.weighting_strategy}")
        print(f"Tasks: {self.n_tasks}")
        print(f"Trees: {len(self.trees)}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Max depth: {self.max_depth}")
        
        if self.track_split_gains and self.iteration_gains:
            print(f"Average iteration gain: {np.mean(self.iteration_gains):.4f}")
            print(f"Total split gains tracked: {len(self.split_gains_history)}")
        
        # Print component summaries
        if hasattr(self.gradient_computer, 'print_summary'):
            self.gradient_computer.print_summary()
        
        if hasattr(self.weighting_manager, 'print_summary'):
            self.weighting_manager.print_summary()
