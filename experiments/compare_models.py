"""
MT-GBM性能比較実験モジュール

このモジュールは、3種類のMT-GBM実装（スクラッチGBDT、XGBoost、LightGBM）の
性能を比較するための実験スクリプトを提供します。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from typing import Dict, List, Tuple, Union, Optional, Any
import json

from ..models.base import MTGBMBase
from ..models.gbdt import MTGBDT
from ..models.xgboost_wrapper import MTXGBoost
from ..models.lgbm_wrapper import MTLightGBM
from ..data.synthetic import generate_synthetic_data, generate_nonlinear_synthetic_data
from ..data.real_data import load_california_housing, load_diabetes_multitask, load_wine_multitask


def run_model_comparison(dataset_name: str,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        n_estimators: int = 100,
                        learning_rate: float = 0.1,
                        max_depth: int = 3,
                        random_state: int = 42,
                        output_dir: str = "results") -> Dict:
    """
    3種類のMT-GBMモデルを比較
    
    Parameters:
    -----------
    dataset_name : str
        データセット名
    X_train : array-like
        訓練用入力特徴量
    y_train : array-like
        訓練用ターゲット値
    X_test : array-like
        テスト用入力特徴量
    y_test : array-like
        テスト用ターゲット値
    n_estimators : int, default=100
        ブースティング反復回数（木の数）
    learning_rate : float, default=0.1
        学習率（各木の寄与度）
    max_depth : int, default=3
        各木の最大深さ
    random_state : int, default=42
        乱数シード
    output_dir : str, default="results"
        結果の出力ディレクトリ
        
    Returns:
    --------
    results : dict
        比較結果
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 共通パラメータ
    common_params = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'random_state': random_state
    }
    
    # 結果を格納する辞書
    results = {
        'dataset': dataset_name,
        'n_samples_train': X_train.shape[0],
        'n_samples_test': X_test.shape[0],
        'n_features': X_train.shape[1],
        'n_tasks': y_train.shape[1],
        'models': {}
    }
    
    # 各モデルを評価
    models = {
        'MTGBDT': MTGBDT(**common_params),
        'MTXGBoost': MTXGBoost(**common_params),
        'MTLightGBM': MTLightGBM(**common_params)
    }
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} on {dataset_name}...")
        
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
        
        # 結果を格納
        results['models'][model_name] = {
            'train_time': train_time,
            'predict_time': predict_time,
            'evaluation': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in eval_results.items()}
        }
        
        # 結果を表示
        print(f"  Train time: {train_time:.4f}s")
        print(f"  Predict time: {predict_time:.4f}s")
        print(f"  MSE: {eval_results['mse_avg']:.6f}")
        print(f"  RMSE: {eval_results['rmse_avg']:.6f}")
        print(f"  MAE: {eval_results['mae_avg']:.6f}")
        print(f"  R2: {eval_results['r2_avg']:.6f}")
    
    # 結果をJSONファイルに保存
    with open(os.path.join(output_dir, f"{dataset_name}_comparison.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 結果をプロット
    plot_comparison_results(results, os.path.join(output_dir, f"{dataset_name}_comparison.png"))
    
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
    model_names = list(results['models'].keys())
    
    # 訓練時間
    train_times = [results['models'][model]['train_time'] for model in model_names]
    
    # 予測時間
    predict_times = [results['models'][model]['predict_time'] for model in model_names]
    
    # MSE
    mse_values = [results['models'][model]['evaluation']['mse_avg'] for model in model_names]
    
    # R2
    r2_values = [results['models'][model]['evaluation']['r2_avg'] for model in model_names]
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # タイトル
    fig.suptitle(f"MT-GBM Model Comparison on {results['dataset']} Dataset", fontsize=16)
    
    # 訓練時間
    sns.barplot(x=model_names, y=train_times, ax=axes[0, 0])
    axes[0, 0].set_title('Training Time (s)')
    axes[0, 0].set_ylabel('Time (s)')
    
    # 予測時間
    sns.barplot(x=model_names, y=predict_times, ax=axes[0, 1])
    axes[0, 1].set_title('Prediction Time (s)')
    axes[0, 1].set_ylabel('Time (s)')
    
    # MSE
    sns.barplot(x=model_names, y=mse_values, ax=axes[1, 0])
    axes[1, 0].set_title('Mean Squared Error (MSE)')
    axes[1, 0].set_ylabel('MSE')
    
    # R2
    sns.barplot(x=model_names, y=r2_values, ax=axes[1, 1])
    axes[1, 1].set_title('R² Score')
    axes[1, 1].set_ylabel('R²')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_task_specific_results(results: Dict, save_path: Optional[str] = None) -> None:
    """
    タスク別の比較結果をプロット
    
    Parameters:
    -----------
    results : dict
        比較結果
    save_path : str, optional
        保存先のパス
    """
    # モデル名
    model_names = list(results['models'].keys())
    
    # タスク数
    n_tasks = len(results['models'][model_names[0]]['evaluation']['mse'])
    
    # プロット
    fig, axes = plt.subplots(n_tasks, 2, figsize=(14, 5 * n_tasks))
    
    # タイトル
    fig.suptitle(f"Task-Specific Performance on {results['dataset']} Dataset", fontsize=16)
    
    for task_idx in range(n_tasks):
        # MSE
        mse_values = [results['models'][model]['evaluation']['mse'][task_idx] for model in model_names]
        
        # R2
        r2_values = [results['models'][model]['evaluation']['r2'][task_idx] for model in model_names]
        
        # MSEプロット
        if n_tasks > 1:
            ax_mse = axes[task_idx, 0]
            ax_r2 = axes[task_idx, 1]
        else:
            ax_mse = axes[0]
            ax_r2 = axes[1]
        
        sns.barplot(x=model_names, y=mse_values, ax=ax_mse)
        ax_mse.set_title(f'Task {task_idx+1} - Mean Squared Error (MSE)')
        ax_mse.set_ylabel('MSE')
        
        # R2プロット
        sns.barplot(x=model_names, y=r2_values, ax=ax_r2)
        ax_r2.set_title(f'Task {task_idx+1} - R² Score')
        ax_r2.set_ylabel('R²')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def run_all_experiments(output_dir: str = "results", random_state: int = 42) -> None:
    """
    すべての実験を実行
    
    Parameters:
    -----------
    output_dir : str, default="results"
        結果の出力ディレクトリ
    random_state : int, default=42
        乱数シード
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 実験設定
    experiments = [
        {
            'name': 'synthetic_linear',
            'data_func': generate_synthetic_data,
            'params': {
                'n_samples': 1000,
                'n_features': 10,
                'n_tasks': 2,
                'task_correlation': 0.5,
                'random_state': random_state
            }
        },
        {
            'name': 'synthetic_nonlinear',
            'data_func': generate_nonlinear_synthetic_data,
            'params': {
                'n_samples': 1000,
                'n_features': 10,
                'n_tasks': 2,
                'task_correlation': 0.5,
                'random_state': random_state
            }
        },
        {
            'name': 'california_housing',
            'data_func': load_california_housing,
            'params': {
                'random_state': random_state
            }
        },
        {
            'name': 'diabetes',
            'data_func': load_diabetes_multitask,
            'params': {
                'random_state': random_state
            }
        },
        {
            'name': 'wine',
            'data_func': load_wine_multitask,
            'params': {
                'random_state': random_state
            }
        }
    ]
    
    # 各実験を実行
    all_results = {}
    
    for exp in experiments:
        print(f"\n\n{'='*50}")
        print(f"Running experiment: {exp['name']}")
        print(f"{'='*50}")
        
        # データを生成/読み込み
        X_train, y_train, X_test, y_test = exp['data_func'](**exp['params'])
        
        # モデル比較を実行
        results = run_model_comparison(
            dataset_name=exp['name'],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_state,
            output_dir=output_dir
        )
        
        # タスク別の結果をプロット
        plot_task_specific_results(
            results,
            save_path=os.path.join(output_dir, f"{exp['name']}_task_specific.png")
        )
        
        all_results[exp['name']] = results
    
    # すべての実験の要約を作成
    summary = {
        'datasets': [],
        'avg_mse': {},
        'avg_r2': {},
        'avg_train_time': {},
        'avg_predict_time': {}
    }
    
    for dataset_name, result in all_results.items():
        summary['datasets'].append(dataset_name)
        
        for model_name in result['models']:
            if model_name not in summary['avg_mse']:
                summary['avg_mse'][model_name] = []
                summary['avg_r2'][model_name] = []
                summary['avg_train_time'][model_name] = []
                summary['avg_predict_time'][model_name] = []
            
            summary['avg_mse'][model_name].append(result['models'][model_name]['evaluation']['mse_avg'])
            summary['avg_r2'][model_name].append(result['models'][model_name]['evaluation']['r2_avg'])
            summary['avg_train_time'][model_name].append(result['models'][model_name]['train_time'])
            summary['avg_predict_time'][model_name].append(result['models'][model_name]['predict_time'])
    
    # 要約をJSONファイルに保存
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 要約をプロット
    plot_summary(summary, os.path.join(output_dir, "summary.png"))


def plot_summary(summary: Dict, save_path: Optional[str] = None) -> None:
    """
    実験の要約をプロット
    
    Parameters:
    -----------
    summary : dict
        実験の要約
    save_path : str, optional
        保存先のパス
    """
    # モデル名
    model_names = list(summary['avg_mse'].keys())
    
    # データセット名
    dataset_names = summary['datasets']
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # タイトル
    fig.suptitle("MT-GBM Models Performance Summary", fontsize=16)
    
    # MSE
    mse_df = pd.DataFrame({
        model: summary['avg_mse'][model] for model in model_names
    }, index=dataset_names)
    
    sns.heatmap(mse_df, annot=True, fmt=".4f", cmap="YlGnBu", ax=axes[0, 0])
    axes[0, 0].set_title('Mean Squared Error (MSE)')
    
    # R2
    r2_df = pd.DataFrame({
        model: summary['avg_r2'][model] for model in model_names
    }, index=dataset_names)
    
    sns.heatmap(r2_df, annot=True, fmt=".4f", cmap="YlGnBu", ax=axes[0, 1])
    axes[0, 1].set_title('R² Score')
    
    # 訓練時間
    train_time_df = pd.DataFrame({
        model: summary['avg_train_time'][model] for model in model_names
    }, index=dataset_names)
    
    sns.heatmap(train_time_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[1, 0])
    axes[1, 0].set_title('Training Time (s)')
    
    # 予測時間
    predict_time_df = pd.DataFrame({
        model: summary['avg_predict_time'][model] for model in model_names
    }, index=dataset_names)
    
    sns.heatmap(predict_time_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[1, 1])
    axes[1, 1].set_title('Prediction Time (s)')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


if __name__ == "__main__":
    # すべての実験を実行
    run_all_experiments(output_dir="results", random_state=42)
