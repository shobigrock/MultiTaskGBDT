"""
Data Transform Utilities

This module contains utility functions for data transformation
and weight computation used in multi-task learning.
"""

import numpy as np
from typing import Optional


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
    epsilon : float, default=1e-3
        Small value to avoid division by zero
    
    Returns:
    --------
    ips_weights : array-like, shape=(n_samples,)
        IPS weights = 1 / max(ctr_pred, epsilon)
    """
    return 1.0 / np.maximum(ctr_pred, epsilon)


def compute_normalization_weights(
    data: np.ndarray, 
    target_mean: float, 
    target_std: float, 
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    データを指定された平均と標準偏差に正規化する重み付けを計算
    
    Parameters:
    -----------
    data : array-like, shape=(n_samples, n_tasks)
        正規化対象のデータ
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


def validate_input_data(X: np.ndarray, y_multi: Optional[np.ndarray] = None) -> tuple:
    """
    入力データの検証と前処理
    
    Parameters:
    -----------
    X : array-like, shape=(n_samples, n_features)
        特徴量行列
    y_multi : array-like, shape=(n_samples, n_tasks), optional
        ターゲット行列
        
    Returns:
    --------
    X_validated : array-like, shape=(n_samples, n_features)
        検証済み特徴量行列
    y_validated : array-like, shape=(n_samples, n_tasks) or None
        検証済みターゲット行列
    """
    # 特徴量の検証
    X = np.array(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got {X.ndim}D")
    
    # ターゲットの検証
    if y_multi is not None:
        y_multi = np.array(y_multi, dtype=np.float64)
        if y_multi.ndim != 2:
            raise ValueError(f"y_multi must be 2D array, got {y_multi.ndim}D")
        if X.shape[0] != y_multi.shape[0]:
            raise ValueError(f"X and y_multi must have same number of samples, got {X.shape[0]} and {y_multi.shape[0]}")
    
    # 無限値とNaN値のチェック
    if np.any(np.isinf(X)) or np.any(np.isnan(X)):
        raise ValueError("X contains inf or NaN values")
    
    return X, y_multi


def _validate_strategy_n_tasks(weighting_strategy: str, n_tasks: int) -> None:
    """
    Validate that the weighting strategy is compatible with the number of tasks
    
    Parameters:
    -----------
    weighting_strategy : str
        The weighting strategy to validate
    n_tasks : int
        The number of tasks
        
    Raises:
    -------
    ValueError
        If the strategy is not compatible with the number of tasks
    """
    strategy_requirements = {
        "mtgbm-three": 3,
        "mtgbm-three-afterIPS": 3,
        "mtgbm-ctr-cvr": 3,
        "cvr-ips-3": 3,
        "cvr-dr-3": 3,
        "proposed-ctcvr-3": 3,
        "proposed-cvr-3": 3
    }
    
    if weighting_strategy in strategy_requirements:
        required_tasks = strategy_requirements[weighting_strategy]
        if n_tasks != required_tasks:
            raise ValueError(f"Strategy '{weighting_strategy}' requires {required_tasks} tasks, but got {n_tasks}")


def _normalize_array(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Normalize array by subtracting mean and dividing by standard deviation
    
    Parameters:
    -----------
    arr : np.ndarray
        Array to normalize
    axis : int
        Axis along which to normalize
        
    Returns:
    --------
    normalized : np.ndarray
        Normalized array
    """
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)  # Avoid division by zero
    return (arr - mean) / std


def _clip_probabilities(probs: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """
    Clip probabilities to avoid numerical issues
    
    Parameters:
    -----------
    probs : np.ndarray
        Probabilities to clip
    epsilon : float
        Minimum and maximum values for clipping
        
    Returns:
    --------
    clipped : np.ndarray
        Clipped probabilities
    """
    return np.clip(probs, epsilon, 1 - epsilon)


def _compute_dr_weights(ctr_pred: np.ndarray, cvr_direct_pred: np.ndarray,
                       actual_click: np.ndarray, actual_cvr: np.ndarray,
                       epsilon: float = 1e-6) -> np.ndarray:
    """
    Doubly Robust (DR) weights for CVR task
    
    DR estimator: E[Y|X,T=1] / P(T=1|X) + (1 - T/P(T=1|X)) * μ(X)
    where μ(X) is the direct regression estimate of CVR
    
    Parameters:
    -----------
    ctr_pred : np.ndarray
        CTR predictions (propensity scores)
    cvr_direct_pred : np.ndarray
        Direct CVR predictions
    actual_click : np.ndarray
        Actual click values (0 or 1)
    actual_cvr : np.ndarray
        Actual CVR values (valid only when click=1)
    epsilon : float
        Small value to avoid division by zero
        
    Returns:
    --------
    dr_weights : np.ndarray
        DR correction weights
    """
    # Clip CTR predictions to avoid numerical issues
    ctr_pred_clipped = np.clip(ctr_pred, epsilon, 1 - epsilon)
    
    # IPS term: Y * T / P(T=1|X) for clicked samples
    ips_term = np.where(actual_click == 1, 
                       actual_cvr / ctr_pred_clipped, 
                       0)
    
    # Direct estimation term: (1 - T/P(T=1|X)) * μ(X)
    direct_term = (1 - actual_click / ctr_pred_clipped) * cvr_direct_pred
    
    # DR weights = IPS term + Direct term
    dr_weights = ips_term + direct_term
    
    return dr_weights


def _get_cvr_direct_estimates(model, X: np.ndarray) -> np.ndarray:
    """
    Get direct CVR estimates from the current model state
    
    Parameters:
    -----------
    model : MTGBDT
        Current model instance
    X : np.ndarray
        Input features
        
    Returns:
    --------
    cvr_estimates : np.ndarray
        Direct CVR probability estimates
    """
    try:
        # Get current predictions from the model
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            if probs.shape[1] >= 3:
                # For 3-task models, use CVR task (index 2)
                return probs[:, 2]
            else:
                # For 2-task models, use a simple baseline
                return np.full(X.shape[0], 0.1)  # Simple baseline
        else:
            # Fallback: use simple baseline
            return np.full(X.shape[0], 0.1)
    except:
        # Fallback: use simple baseline
        return np.full(X.shape[0], 0.1)
