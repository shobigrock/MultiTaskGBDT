"""
Weighting Strategies Manager

This module contains all weighting strategies for multi-task learning
and manages the strategy selection and execution.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from .data_transforms import _compute_ips_weight, compute_normalization_weights


class WeightingStrategyManager:
    """
    重み付け戦略を管理するクラス
    
    Attributes:
    -----------
    strategy : str
        使用する重み付け戦略
    n_tasks : int
        タスク数
    gamma : float
        重み付けパラメータ
    delta : float
        重み付けパラメータ
    """
    
    def __init__(self, weighting_strategy: str = "mtgbm", n_tasks: int = 2, gamma: float = 50.0, 
                 delta: float = 0.5, gain_threshold: float = 0.1, is_dynamic_weight: bool = False):
        self.strategy = weighting_strategy
        self.n_tasks = n_tasks
        self.gamma = gamma
        self.delta = delta
        self.gain_threshold = gain_threshold
        self.is_dynamic_weight = is_dynamic_weight
        self.n_tasks = n_tasks
        self.gamma = gamma
        self.delta = delta
        
        # 利用可能な戦略
        self.available_strategies = {
            "mtgbm": self._compute_mtgbm_weights,
            "mtgbm-three": self._compute_mtgbm_three_weights,
            "mtgbm-ctr-cvr": self._compute_mtgbm_ctr_cvr_weights,
            "mtgbm-three-afterIPS": self._compute_mtgbm_three_afterIPS_weights,
            "cvr-ips-3": self._compute_cvr_ips_3_weights,
            "cvr-dr-3": self._compute_cvr_dr_3_weights,
            "proposed": self._compute_proposed_weights,
            "proposed_reverse": self._compute_proposed_reverse_weights,
            "half": self._compute_half_weights
        }
    
    def compute_ensemble_weights_gradients_hessians(
        self,
        raw_gradients: np.ndarray,
        raw_hessians: np.ndarray,
        task_correlations: np.ndarray,
        current_predictions: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        アンサンブル重み付けされた勾配とヘシアンを計算
        
        Parameters:
        -----------
        raw_gradients : array-like, shape=(n_samples, n_tasks)
            生の勾配
        raw_hessians : array-like, shape=(n_samples, n_tasks)
            生のヘシアン
        task_correlations : array-like, shape=(n_tasks, n_tasks)
            タスク間相関行列
        current_predictions : array-like, shape=(n_samples, n_tasks), optional
            現在の予測値
        y_true : array-like, shape=(n_samples, n_tasks), optional
            真のターゲット値
            
        Returns:
        --------
        ensemble_gradients : array-like, shape=(n_samples, n_tasks)
            アンサンブル重み付け勾配
        ensemble_hessians : array-like, shape=(n_samples, n_tasks)
            アンサンブル重み付けヘシアン
        """
        if self.strategy not in self.available_strategies:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return self.available_strategies[self.strategy](
            raw_gradients, raw_hessians, task_correlations, current_predictions, y_true
        )
    
    def _compute_mtgbm_weights(
        self,
        raw_gradients: np.ndarray,
        raw_hessians: np.ndarray,
        task_correlations: np.ndarray,
        current_predictions: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """元のMT-GBM戦略"""
        n_samples, n_tasks = raw_gradients.shape
        ensemble_gradients = np.zeros_like(raw_gradients)
        ensemble_hessians = np.zeros_like(raw_hessians)
        
        # 各タスクについて重み付け
        for task_idx in range(n_tasks):
            task_weight = 1.0 / n_tasks  # 均等重み
            ensemble_gradients[:, task_idx] = task_weight * raw_gradients[:, task_idx]
            ensemble_hessians[:, task_idx] = task_weight * raw_hessians[:, task_idx]
        
        return ensemble_gradients, ensemble_hessians
    
    def _compute_mtgbm_three_weights(
        self,
        raw_gradients: np.ndarray,
        raw_hessians: np.ndarray,
        task_correlations: np.ndarray,
        current_predictions: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """3タスク専用MT-GBM戦略"""
        n_samples, n_tasks = raw_gradients.shape
        ensemble_gradients = np.zeros_like(raw_gradients)
        ensemble_hessians = np.zeros_like(raw_hessians)
        
        # 3タスクの場合の重み設定
        if n_tasks == 3:
            # CTR: 0.4, CTCVR: 0.4, CVR: 0.2
            weights = [0.4, 0.4, 0.2]
        else:
            # デフォルトは均等重み
            weights = [1.0 / n_tasks] * n_tasks
        
        for task_idx in range(n_tasks):
            ensemble_gradients[:, task_idx] = weights[task_idx] * raw_gradients[:, task_idx]
            ensemble_hessians[:, task_idx] = weights[task_idx] * raw_hessians[:, task_idx]
        
        return ensemble_gradients, ensemble_hessians
    
    def _compute_mtgbm_ctr_cvr_weights(
        self,
        raw_gradients: np.ndarray,
        raw_hessians: np.ndarray,
        task_correlations: np.ndarray,
        current_predictions: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CTR-CVR専用戦略"""
        n_samples, n_tasks = raw_gradients.shape
        ensemble_gradients = np.zeros_like(raw_gradients)
        ensemble_hessians = np.zeros_like(raw_hessians)
        
        # CTR-CVRに重点を置いた重み設定
        if n_tasks >= 3:
            # CTR: 0.5, CTCVR: 0.3, CVR: 0.2
            weights = [0.5, 0.3, 0.2]
        else:
            weights = [1.0 / n_tasks] * n_tasks
        
        for task_idx in range(min(n_tasks, len(weights))):
            ensemble_gradients[:, task_idx] = weights[task_idx] * raw_gradients[:, task_idx]
            ensemble_hessians[:, task_idx] = weights[task_idx] * raw_hessians[:, task_idx]
        
        return ensemble_gradients, ensemble_hessians
    
    def _compute_mtgbm_three_afterIPS_weights(
        self,
        raw_gradients: np.ndarray,
        raw_hessians: np.ndarray,
        task_correlations: np.ndarray,
        current_predictions: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """正規化後IPS重み付け戦略"""
        n_samples, n_tasks = raw_gradients.shape
        
        # まず正規化を実行
        valid_mask = ~np.isnan(raw_gradients)
        normalized_gradients = compute_normalization_weights(
            raw_gradients, target_mean=0.0, target_std=1.0, mask=valid_mask
        )
        normalized_hessians = compute_normalization_weights(
            raw_hessians, target_mean=0.25, target_std=0.1, mask=valid_mask
        )
        
        # IPS重み付けを適用（CVRタスクのみ）
        if n_tasks >= 3 and current_predictions is not None:
            # CTR予測値を使用してIPS重みを計算
            ctr_predictions = 1.0 / (1.0 + np.exp(-current_predictions[:, 0]))
            ips_weights = _compute_ips_weight(ctr_predictions)
            
            # CVRタスク（タスク2）にIPS重みを適用
            cvr_mask = ~np.isnan(raw_gradients[:, 2])
            normalized_gradients[cvr_mask, 2] *= ips_weights[cvr_mask]
            normalized_hessians[cvr_mask, 2] *= ips_weights[cvr_mask]
        
        return normalized_gradients, normalized_hessians
    
    def _compute_proposed_weights(
        self,
        raw_gradients: np.ndarray,
        raw_hessians: np.ndarray,
        task_correlations: np.ndarray,
        current_predictions: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """提案手法の動的重み付け"""
        n_samples, n_tasks = raw_gradients.shape
        ensemble_gradients = np.zeros_like(raw_gradients)
        ensemble_hessians = np.zeros_like(raw_hessians)
        
        # タスク間相関を使用した動的重み計算
        for task_idx in range(n_tasks):
            # 他のタスクとの相関を考慮した重み
            correlation_sum = np.sum(np.abs(task_correlations[task_idx, :]))
            if correlation_sum > 0:
                task_weight = 1.0 / correlation_sum
            else:
                task_weight = 1.0 / n_tasks
            
            ensemble_gradients[:, task_idx] = task_weight * raw_gradients[:, task_idx]
            ensemble_hessians[:, task_idx] = task_weight * raw_hessians[:, task_idx]
        
        return ensemble_gradients, ensemble_hessians
    
    def _compute_proposed_reverse_weights(
        self,
        raw_gradients: np.ndarray,
        raw_hessians: np.ndarray,
        task_correlations: np.ndarray,
        current_predictions: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """提案手法の逆重み付け"""
        n_samples, n_tasks = raw_gradients.shape
        ensemble_gradients = np.zeros_like(raw_gradients)
        ensemble_hessians = np.zeros_like(raw_hessians)
        
        # タスク間相関を使用した逆重み計算
        for task_idx in range(n_tasks):
            # 他のタスクとの相関が高いほど重みを大きく
            correlation_sum = np.sum(np.abs(task_correlations[task_idx, :]))
            if correlation_sum > 0:
                task_weight = correlation_sum / n_tasks
            else:
                task_weight = 1.0 / n_tasks
            
            ensemble_gradients[:, task_idx] = task_weight * raw_gradients[:, task_idx]
            ensemble_hessians[:, task_idx] = task_weight * raw_hessians[:, task_idx]
        
        return ensemble_gradients, ensemble_hessians
    
    def _compute_half_weights(
        self,
        raw_gradients: np.ndarray,
        raw_hessians: np.ndarray,
        task_correlations: np.ndarray,
        current_predictions: Optional[np.ndarray] = None,
        y_true: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """固定0.5重み付け"""
        n_samples, n_tasks = raw_gradients.shape
        ensemble_gradients = np.zeros_like(raw_gradients)
        ensemble_hessians = np.zeros_like(raw_hessians)
        
        # 全タスクに0.5の重み（2タスクの場合は等価）
        for task_idx in range(n_tasks):
            ensemble_gradients[:, task_idx] = 0.5 * raw_gradients[:, task_idx]
            ensemble_hessians[:, task_idx] = 0.5 * raw_hessians[:, task_idx]
        
        return ensemble_gradients, ensemble_hessians
    
    def _compute_cvr_ips_3_weights(self, gradients: np.ndarray, hessians: np.ndarray,
                                   task_correlations: np.ndarray, information_gain: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        CVR-IPS-3戦略: CVRタスクをIPS重みで強調し、情報利得が低い場合はCTRタスクを強調
        
        Parameters:
        -----------
        gradients : np.ndarray
            勾配
        hessians : np.ndarray
            ヘシアン
        task_correlations : np.ndarray
            タスク間相関
        information_gain : float
            現在の情報利得
            
        Returns:
        --------
        ensemble_gradients : np.ndarray
            アンサンブル勾配
        ensemble_hessians : np.ndarray
            アンサンブルヘシアン
        """
        if self.n_tasks != 3:
            raise ValueError("cvr-ips-3 strategy requires n_tasks=3")
        
        # 情報利得の閾値（デフォルト0.1）
        gain_threshold = getattr(self, 'gain_threshold', 0.1)
        
        if information_gain < gain_threshold:
            # 情報利得が低い場合：CTRタスクを強調
            weights = np.array([0.8, 0.1, 0.1])  # CTR強調
        else:
            # 通常：CVRタスクを強調（IPS重みは既に適用済み）
            weights = np.array([0.2, 0.2, 0.6])  # CVR強調
        
        # アンサンブル計算
        ensemble_gradients = np.zeros_like(gradients)
        ensemble_hessians = np.zeros_like(hessians)
        
        for task_idx in range(self.n_tasks):
            ensemble_gradients[:, task_idx] = weights[task_idx] * gradients[:, task_idx]
            ensemble_hessians[:, task_idx] = weights[task_idx] * hessians[:, task_idx]
        
        return ensemble_gradients, ensemble_hessians
    
    def _compute_cvr_dr_3_weights(self, gradients: np.ndarray, hessians: np.ndarray,
                                  task_correlations: np.ndarray, information_gain: float = 0.0,
                                  model=None, X=None, y_multi=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        CVR-DR-3戦略: CVRタスクにDoubly Robust補正を適用
        
        Parameters:
        -----------
        gradients : np.ndarray
            勾配
        hessians : np.ndarray
            ヘシアン
        task_correlations : np.ndarray
            タスク間相関
        information_gain : float
            現在の情報利得
        model : MTGBDT, optional
            現在のモデル（CVR直接推定用）
        X : np.ndarray, optional
            入力特徴量
        y_multi : np.ndarray, optional
            ターゲット値
            
        Returns:
        --------
        ensemble_gradients : np.ndarray
            アンサンブル勾配
        ensemble_hessians : np.ndarray
            アンサンブルヘシアン
        """
        if self.n_tasks != 3:
            raise ValueError("cvr-dr-3 strategy requires n_tasks=3")
        
        ensemble_gradients = gradients.copy()
        ensemble_hessians = hessians.copy()
        
        # DR補正を適用（CVRタスクのみ）
        if model is not None and X is not None and y_multi is not None:
            from .data_transforms import _compute_dr_weights, _get_cvr_direct_estimates
            
            # CTR予測値を取得
            try:
                ctr_pred = model.predict_proba(X)[:, 0]
            except:
                # フォールバック：平均CTR率を使用
                ctr_pred = np.full(X.shape[0], np.mean(y_multi[:, 0]))
            
            # CVR直接推定値を取得
            cvr_direct_pred = _get_cvr_direct_estimates(model, X)
            
            # DR重みを計算
            dr_weights = _compute_dr_weights(
                ctr_pred, cvr_direct_pred, y_multi[:, 0], y_multi[:, 2]
            )
            
            # CVRタスクの勾配・ヘシアンに適用
            valid_mask = ~np.isnan(y_multi[:, 2])
            ensemble_gradients[valid_mask, 2] *= dr_weights[valid_mask]
            ensemble_hessians[valid_mask, 2] *= dr_weights[valid_mask]
        
        return ensemble_gradients, ensemble_hessians
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        現在の戦略情報を取得
        
        Returns:
        --------
        info : dict
            戦略情報
        """
        return {
            "strategy": self.strategy,
            "n_tasks": self.n_tasks,
            "gamma": self.gamma,
            "delta": self.delta,
            "available_strategies": list(self.available_strategies.keys())
        }
