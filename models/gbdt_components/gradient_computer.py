"""
Gradient Computer

This module handles gradient and hessian computation for multi-task learning
with various loss functions and normalization strategies.
"""

import numpy as np
from typing import Tuple, Optional
from .data_transforms import compute_normalization_weights


class GradientComputer:
    """
    勾配とヘシアンの計算を担当するクラス
    
    Attributes:
    -----------
    loss : str
        損失関数の種類
    n_tasks : int
        タスク数
    """
    
    def __init__(self, loss: str = "logloss", n_tasks: int = 2, 
                 normalize_gradients: bool = False, weighting_strategy: str = "mtgbm"):
        self.loss = loss
        self.n_tasks = n_tasks
        self.normalize_gradients = normalize_gradients
        self.weighting_strategy = weighting_strategy
        
    def compute_initial_predictions(self, y_multi: np.ndarray) -> np.ndarray:
        """
        初期予測値を計算
        
        Parameters:
        -----------
        y_multi : array-like, shape=(n_samples, n_tasks)
            ターゲット値（マルチタスク）
            
        Returns:
        --------
        initial_preds : array-like, shape=(n_tasks,)
            各タスクの初期予測値
        """
        initial_preds = np.zeros(self.n_tasks)
        
        for task_idx in range(self.n_tasks):
            task_targets = y_multi[:, task_idx]
            
            # NaN値を除外して平均を計算
            valid_mask = ~np.isnan(task_targets)
            if np.any(valid_mask):
                task_mean = np.mean(task_targets[valid_mask])
                # ロジット変換
                task_mean = np.clip(task_mean, 1e-7, 1 - 1e-7)
                initial_preds[task_idx] = np.log(task_mean / (1 - task_mean))
            else:
                initial_preds[task_idx] = 0.0
                
        return initial_preds
    
    def compute_gradients_hessians(
        self, 
        y_multi: np.ndarray, 
        y_pred_logits: np.ndarray, 
        iteration: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        勾配とヘシアンを計算
        
        Parameters:
        -----------
        y_multi : array-like, shape=(n_samples, n_tasks)
            真のターゲット値
        y_pred_logits : array-like, shape=(n_samples, n_tasks)
            予測ロジット値
        iteration : int, default=0
            現在の反復回数
            
        Returns:
        --------
        gradients : array-like, shape=(n_samples, n_tasks)
            勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            ヘシアン
        """
        if self.loss == "logloss":
            return self._compute_logloss_gradients_hessians(y_multi, y_pred_logits)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")
    
    def _compute_logloss_gradients_hessians(
        self, 
        y_multi: np.ndarray, 
        y_pred_logits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ログ損失の勾配とヘシアンを計算
        
        Parameters:
        -----------
        y_multi : array-like, shape=(n_samples, n_tasks)
            真のターゲット値
        y_pred_logits : array-like, shape=(n_samples, n_tasks)
            予測ロジット値
            
        Returns:
        --------
        gradients : array-like, shape=(n_samples, n_tasks)
            勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            ヘシアン
        """
        n_samples, n_tasks = y_multi.shape
        gradients = np.zeros((n_samples, n_tasks))
        hessians = np.zeros((n_samples, n_tasks))
        
        for task_idx in range(n_tasks):
            y_true = y_multi[:, task_idx]
            y_pred_logit = y_pred_logits[:, task_idx]
            
            # 有効なデータのマスク（NaNでない）
            valid_mask = ~np.isnan(y_true)
            
            if np.any(valid_mask):
                # シグモイド関数の適用
                y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred_logit[valid_mask]))
                
                # 勾配: pred_prob - true_label
                gradients[valid_mask, task_idx] = y_pred_prob - y_true[valid_mask]
                
                # ヘシアン: pred_prob * (1 - pred_prob)
                hessians[valid_mask, task_idx] = y_pred_prob * (1 - y_pred_prob)
                
                # 無効データは勾配とヘシアンを0に設定
                gradients[~valid_mask, task_idx] = 0.0
                hessians[~valid_mask, task_idx] = 0.0
            else:
                # 全てのデータが無効な場合
                gradients[:, task_idx] = 0.0
                hessians[:, task_idx] = 0.0
        
        return gradients, hessians
    
    def compute_task_correlations(self, gradients: np.ndarray) -> np.ndarray:
        """
        タスク間の相関を計算
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            勾配
            
        Returns:
        --------
        correlations : array-like, shape=(n_tasks, n_tasks)
            タスク間相関行列
        """
        n_tasks = gradients.shape[1]
        correlations = np.zeros((n_tasks, n_tasks))
        
        for i in range(n_tasks):
            for j in range(n_tasks):
                if i == j:
                    correlations[i, j] = 1.0
                else:
                    # 有効なデータのマスク
                    valid_mask_i = ~np.isnan(gradients[:, i])
                    valid_mask_j = ~np.isnan(gradients[:, j])
                    valid_mask = valid_mask_i & valid_mask_j
                    
                    if np.sum(valid_mask) > 1:
                        grad_i = gradients[valid_mask, i]
                        grad_j = gradients[valid_mask, j]
                        
                        # 相関係数の計算
                        if np.std(grad_i) > 1e-8 and np.std(grad_j) > 1e-8:
                            correlation = np.corrcoef(grad_i, grad_j)[0, 1]
                            correlations[i, j] = correlation if not np.isnan(correlation) else 0.0
                        else:
                            correlations[i, j] = 0.0
                    else:
                        correlations[i, j] = 0.0
        
        return correlations
    
    def normalize_gradients_hessians(
        self, 
        gradients: np.ndarray, 
        hessians: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        勾配とヘシアンを正規化
        
        Parameters:
        -----------
        gradients : array-like, shape=(n_samples, n_tasks)
            勾配
        hessians : array-like, shape=(n_samples, n_tasks)
            ヘシアン
            
        Returns:
        --------
        norm_gradients : array-like, shape=(n_samples, n_tasks)
            正規化後の勾配
        norm_hessians : array-like, shape=(n_samples, n_tasks)
            正規化後のヘシアン
        """
        # 有効データのマスクを作成
        valid_mask = ~np.isnan(gradients)
        
        # 勾配の正規化
        norm_gradients = compute_normalization_weights(
            gradients, 
            target_mean=0.0, 
            target_std=1.0, 
            mask=valid_mask
        )
        
        # ヘシアンの正規化
        norm_hessians = compute_normalization_weights(
            hessians, 
            target_mean=0.25, 
            target_std=0.1, 
            mask=valid_mask
        )
        
        return norm_gradients, norm_hessians
