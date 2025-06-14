"""
モデルインターフェース統一テスト用モジュール

このモジュールは、MT-GBMの各実装（スクラッチGBDT、XGBoost、LightGBM）の
共通インターフェースをテストするためのユーティリティを提供します。
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import time
import matplotlib.pyplot as plt
from ..models.base import MTGBMBase
from ..models.gbdt import MTGBDT
from ..models.xgboost_wrapper import MTXGBoost
from ..models.lgbm_wrapper import MTLightGBM


def generate_simple_data(n_samples: int = 1000, n_features: int = 10, n_tasks: int = 2, 
                         test_size: float = 0.2, random_state: Optional[int] = None) -> Tuple:
    """
    簡単なマルチタスク回帰データを生成
    
    Parameters:
    -----------
    n_samples : int, default=1000
        サンプル数
    n_features : int, default=10
        特徴量の数
    n_tasks : int, default=2
        タスク数
    test_size : float, default=0.2
        テストデータの割合
    random_state : int, optional
        乱数シード
        
    Returns:
    --------
    X_train : array-like, shape=(n_samples * (1 - test_size), n_features)
        訓練用入力特徴量
    y_train : array-like, shape=(n_samples * (1 - test_size), n_tasks)
        訓練用ターゲット値
    X_test : array-like, shape=(n_samples * test_size, n_features)
        テスト用入力特徴量
    y_test : array-like, shape=(n_samples * test_size, n_tasks)
        テスト用ターゲット値
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # 入力特徴量を生成
    X = np.random.randn(n_samples, n_features)
    
    # 各タスクのターゲット値を生成
    y = np.zeros((n_samples, n_tasks))
    
    # 共通の特徴を使用するタスク
    shared_features = np.random.choice(n_features, size=n_features // 2, replace=False)
    
    for i in range(n_tasks):
        # タスク固有の特徴
        task_features = np.random.choice(
            [f for f in range(n_features) if f not in shared_features],
            size=n_features // 4,
            replace=False
        )
        
        # 共通特徴と固有特徴を組み合わせて使用
        used_features = np.concatenate([shared_features, task_features])
        
        # 特徴の重みを生成
        weights = np.random.randn(len(used_features))
        
        # ターゲット値を計算
        for j, feature_idx in enumerate(used_features):
            y[:, i] += weights[j] * X[:, feature_idx]
        
        # ノイズを追加
        y[:, i] += np.random.randn(n_samples) * 0.1
    
    # 訓練データとテストデータに分割
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    return X_train, y_train, X_test, y_test


def test_model_interface(model_class, model_params: Dict = None, 
                         n_samples: int = 1000, n_features: int = 10, n_tasks: int = 2,
                         random_state: int = 42) -> Dict:
    """
    モデルのインターフェースをテスト
    
    Parameters:
    -----------
    model_class : class
        テストするモデルクラス
    model_params : dict, optional
        モデルのパラメータ
    n_samples : int, default=1000
        サンプル数
    n_features : int, default=10
        特徴量の数
    n_tasks : int, default=2
        タスク数
    random_state : int, default=42
        乱数シード
        
    Returns:
    --------
    results : dict
        テスト結果
    """
    # デフォルトパラメータ
    if model_params is None:
        model_params = {
            'n_estimators': 10,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': random_state
        }
    
    # データを生成
    X_train, y_train, X_test, y_test = generate_simple_data(
        n_samples=n_samples,
        n_features=n_features,
        n_tasks=n_tasks,
        random_state=random_state
    )
    
    # モデルを初期化
    model = model_class(**model_params)
    
    # 学習時間を計測
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 予測時間を計測
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    # 評価
    eval_results = model.evaluate(X_test, y_test, metrics=['mse', 'rmse', 'mae', 'r2'])
    
    # 結果を返す
    results = {
        'model_class': model_class.__name__,
        'train_time': train_time,
        'predict_time': predict_time,
        'evaluation': eval_results
    }
    
    return results


def compare_models(n_samples: int = 1000, n_features: int = 10, n_tasks: int = 2,
                  random_state: int = 42) -> Dict:
    """
    各モデルを比較
    
    Parameters:
    -----------
    n_samples : int, default=1000
        サンプル数
    n_features : int, default=10
        特徴量の数
    n_tasks : int, default=2
        タスク数
    random_state : int, default=42
        乱数シード
        
    Returns:
    --------
    results : dict
        比較結果
    """
    # 共通パラメータ
    common_params = {
        'n_estimators': 50,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': random_state
    }
    
    # 各モデルをテスト
    results = {}
    
    # スクラッチGBDT
    print("Testing MTGBDT...")
    results['MTGBDT'] = test_model_interface(
        MTGBDT,
        model_params=common_params,
        n_samples=n_samples,
        n_features=n_features,
        n_tasks=n_tasks,
        random_state=random_state
    )
    
    # XGBoost
    print("Testing MTXGBoost...")
    results['MTXGBoost'] = test_model_interface(
        MTXGBoost,
        model_params=common_params,
        n_samples=n_samples,
        n_features=n_features,
        n_tasks=n_tasks,
        random_state=random_state
    )
    
    # LightGBM
    print("Testing MTLightGBM...")
    results['MTLightGBM'] = test_model_interface(
        MTLightGBM,
        model_params=common_params,
        n_samples=n_samples,
        n_features=n_features,
        n_tasks=n_tasks,
        random_state=random_state
    )
    
    return results


def plot_comparison_results(results: Dict, save_path: Optional[str] = None) -> None:
    """
    比較結果をプロット
    
    Parameters:
    -----------
    results : dict
        比較結果
    save_path : str, optional
        保存先のパス
    """
    # モデル名
    model_names = list(results.keys())
    
    # 訓練時間
    train_times = [results[model]['train_time'] for model in model_names]
    
    # 予測時間
    predict_times = [results[model]['predict_time'] for model in model_names]
    
    # MSE
    mse_values = [results[model]['evaluation']['mse_avg'] for model in model_names]
    
    # R2
    r2_values = [results[model]['evaluation']['r2_avg'] for model in model_names]
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 訓練時間
    axes[0, 0].bar(model_names, train_times)
    axes[0, 0].set_title('Training Time (s)')
    axes[0, 0].set_ylabel('Time (s)')
    
    # 予測時間
    axes[0, 1].bar(model_names, predict_times)
    axes[0, 1].set_title('Prediction Time (s)')
    axes[0, 1].set_ylabel('Time (s)')
    
    # MSE
    axes[1, 0].bar(model_names, mse_values)
    axes[1, 0].set_title('Mean Squared Error (MSE)')
    axes[1, 0].set_ylabel('MSE')
    
    # R2
    axes[1, 1].bar(model_names, r2_values)
    axes[1, 1].set_title('R² Score')
    axes[1, 1].set_ylabel('R²')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


if __name__ == "__main__":
    # モデルを比較
    results = compare_models(n_samples=1000, n_features=10, n_tasks=2, random_state=42)
    
    # 結果をプロット
    plot_comparison_results(results, save_path="model_comparison.png")
