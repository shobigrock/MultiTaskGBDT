"""
マルチタスクGBDT vs 単一タスクGBDT比較実験モジュール

このモジュールは、マルチタスク学習（MT-GBM）と単一タスク学習（各タスク独立GBDT）の
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

# 単一タスク学習用のXGBoostラッパー
import xgboost as xgb

class SingleTaskXGBoost:
    """
    単一タスク学習用のXGBoostラッパー
    """
    
    def __init__(self, 
                 n_estimators: int = 100, 
                 learning_rate: float = 0.1, 
                 max_depth: int = 3,
                 random_state: Optional[int] = None,
                 **kwargs):
        """
        初期化メソッド
        
        Parameters:
        -----------
        n_estimators : int, default=100
            ブースティング反復回数（木の数）
        learning_rate : float, default=0.1
            学習率（各木の寄与度）
        max_depth : int, default=3
            各木の最大深さ
        random_state : int, optional
            乱数シード
        **kwargs : dict
            XGBoostに渡す追加のパラメータ
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.kwargs = kwargs
        self.models = []
        self.n_tasks = None
    
    def fit(self, X: np.ndarray, y_multi: np.ndarray) -> 'SingleTaskXGBoost':
        """
        各タスクに対して独立したモデルを学習
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクのターゲット値
            
        Returns:
        --------
        self : SingleTaskXGBoost
            学習済みモデル
        """
        self.n_tasks = y_multi.shape[1]
        self.models = []
        
        # XGBoostパラメータを設定
        params = {
            'max_depth': self.max_depth,
            'eta': self.learning_rate,
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'verbosity': 0,
        }
        
        if self.random_state is not None:
            params['seed'] = self.random_state
        
        # 追加のパラメータを設定
        params.update(self.kwargs)
        
        # 各タスクに対して独立したモデルを学習
        for task_idx in range(self.n_tasks):
            # 現在のタスクのターゲット値
            y_task = y_multi[:, task_idx]
            
            # DMatrixを作成
            dtrain = xgb.DMatrix(X, label=y_task)
            
            # モデルを学習
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.n_estimators,
                verbose_eval=False
            )
            
            # モデルを保存
            self.models.append(model)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        学習済みモデルで予測
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        y_pred : array-like, shape=(n_samples, n_tasks)
            各タスクの予測値
        """
        if not self.models:
            raise ValueError("Model has not been trained yet")
        
        # DMatrixを作成
        dtest = xgb.DMatrix(X)
        
        # 各タスクの予測値を計算
        y_pred = np.zeros((X.shape[0], self.n_tasks))
        
        for task_idx, model in enumerate(self.models):
            y_pred[:, task_idx] = model.predict(dtest)
        
        return y_pred
    
    def evaluate(self, X: np.ndarray, y_multi: np.ndarray, metrics: List[str] = ['mse']) -> Dict:
        """
        モデルの評価
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクの真のターゲット値
        metrics : list of str, default=['mse']
            使用する評価指標のリスト
            
        Returns:
        --------
        results : dict
            各タスクの各評価指標の値
        """
        # 予測
        y_pred = self.predict(X)
        
        # 結果格納用辞書
        results = {}
        
        # 各評価指標を計算
        for metric in metrics:
            if metric.lower() == 'mse':
                # 平均二乗誤差
                task_mse = np.mean((y_multi - y_pred) ** 2, axis=0)
                results['mse'] = task_mse
                results['mse_avg'] = np.mean(task_mse)
            
            elif metric.lower() == 'rmse':
                # 平方根平均二乗誤差
                task_rmse = np.sqrt(np.mean((y_multi - y_pred) ** 2, axis=0))
                results['rmse'] = task_rmse
                results['rmse_avg'] = np.mean(task_rmse)
            
            elif metric.lower() == 'mae':
                # 平均絶対誤差
                task_mae = np.mean(np.abs(y_multi - y_pred), axis=0)
                results['mae'] = task_mae
                results['mae_avg'] = np.mean(task_mae)
            
            elif metric.lower() == 'r2':
                # 決定係数
                y_mean = np.mean(y_multi, axis=0)
                ss_tot = np.sum((y_multi - y_mean.reshape(1, -1)) ** 2, axis=0)
                ss_res = np.sum((y_multi - y_pred) ** 2, axis=0)
                task_r2 = 1 - (ss_res / ss_tot)
                results['r2'] = task_r2
                results['r2_avg'] = np.mean(task_r2)
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        return results


# 単一タスク学習用のLightGBMラッパー
import lightgbm as lgb

class SingleTaskLightGBM:
    """
    単一タスク学習用のLightGBMラッパー
    """
    
    def __init__(self, 
                 n_estimators: int = 100, 
                 learning_rate: float = 0.1, 
                 max_depth: int = 3,
                 num_leaves: int = 31,
                 random_state: Optional[int] = None,
                 **kwargs):
        """
        初期化メソッド
        
        Parameters:
        -----------
        n_estimators : int, default=100
            ブースティング反復回数（木の数）
        learning_rate : float, default=0.1
            学習率（各木の寄与度）
        max_depth : int, default=3
            各木の最大深さ
        num_leaves : int, default=31
            一つの木の最大葉ノード数
        random_state : int, optional
            乱数シード
        **kwargs : dict
            LightGBMに渡す追加のパラメータ
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.kwargs = kwargs
        self.models = []
        self.n_tasks = None
    
    def fit(self, X: np.ndarray, y_multi: np.ndarray) -> 'SingleTaskLightGBM':
        """
        各タスクに対して独立したモデルを学習
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクのターゲット値
            
        Returns:
        --------
        self : SingleTaskLightGBM
            学習済みモデル
        """
        self.n_tasks = y_multi.shape[1]
        self.models = []
        
        # LightGBMパラメータを設定
        params = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'objective': 'regression',
            'verbose': -1,
        }
        
        if self.random_state is not None:
            params['seed'] = self.random_state
        
        # 追加のパラメータを設定
        params.update(self.kwargs)
        
        # 各タスクに対して独立したモデルを学習
        for task_idx in range(self.n_tasks):
            # 現在のタスクのターゲット値
            y_task = y_multi[:, task_idx]
            
            # データセットを作成
            train_data = lgb.Dataset(X, label=y_task)
            
            # モデルを学習
            try:
                # 新しいバージョンのLightGBMの場合
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=self.n_estimators,
                    callbacks=[lgb.log_evaluation(period=0)]  # ログ出力を無効化
                )
            except TypeError:
                # 古いバージョンのLightGBMの場合の互換性対応
                try:
                    model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=self.n_estimators,
                        verbose_eval=False
                    )
                except:
                    # 最もシンプルな形式
                    model = lgb.train(
                        params,
                        train_data,
                        num_boost_round=self.n_estimators
                    )
            
            # モデルを保存
            self.models.append(model)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        学習済みモデルで予測
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        y_pred : array-like, shape=(n_samples, n_tasks)
            各タスクの予測値
        """
        if not self.models:
            raise ValueError("Model has not been trained yet")
        
        # 各タスクの予測値を計算
        y_pred = np.zeros((X.shape[0], self.n_tasks))
        
        for task_idx, model in enumerate(self.models):
            y_pred[:, task_idx] = model.predict(X)
        
        return y_pred
    
    def evaluate(self, X: np.ndarray, y_multi: np.ndarray, metrics: List[str] = ['mse']) -> Dict:
        """
        モデルの評価
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクの真のターゲット値
        metrics : list of str, default=['mse']
            使用する評価指標のリスト
            
        Returns:
        --------
        results : dict
            各タスクの各評価指標の値
        """
        # 予測
        y_pred = self.predict(X)
        
        # 結果格納用辞書
        results = {}
        
        # 各評価指標を計算
        for metric in metrics:
            if metric.lower() == 'mse':
                # 平均二乗誤差
                task_mse = np.mean((y_multi - y_pred) ** 2, axis=0)
                results['mse'] = task_mse
                results['mse_avg'] = np.mean(task_mse)
            
            elif metric.lower() == 'rmse':
                # 平方根平均二乗誤差
                task_rmse = np.sqrt(np.mean((y_multi - y_pred) ** 2, axis=0))
                results['rmse'] = task_rmse
                results['rmse_avg'] = np.mean(task_rmse)
            
            elif metric.lower() == 'mae':
                # 平均絶対誤差
                task_mae = np.mean(np.abs(y_multi - y_pred), axis=0)
                results['mae'] = task_mae
                results['mae_avg'] = np.mean(task_mae)
            
            elif metric.lower() == 'r2':
                # 決定係数
                y_mean = np.mean(y_multi, axis=0)
                ss_tot = np.sum((y_multi - y_mean.reshape(1, -1)) ** 2, axis=0)
                ss_res = np.sum((y_multi - y_pred) ** 2, axis=0)
                task_r2 = 1 - (ss_res / ss_tot)
                results['r2'] = task_r2
                results['r2_avg'] = np.mean(task_r2)
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        return results


# 単一タスク学習用のスクラッチGBDTラッパー
from ..models.gbdt import MultiTaskDecisionTree

class SingleTaskGBDT:
    """
    単一タスク学習用のスクラッチGBDTラッパー
    """
    
    def __init__(self, 
                 n_estimators: int = 100, 
                 learning_rate: float = 0.1, 
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 random_state: Optional[int] = None):
        """
        初期化メソッド
        
        Parameters:
        -----------
        n_estimators : int, default=100
            ブースティング反復回数（木の数）
        learning_rate : float, default=0.1
            学習率（各木の寄与度）
        max_depth : int, default=3
            各木の最大深さ
        min_samples_split : int, default=2
            分割に必要な最小サンプル数
        min_samples_leaf : int, default=1
            リーフノードに必要な最小サンプル数
        random_state : int, optional
            乱数シード
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.trees = []
        self.initial_predictions = []
        self.n_tasks = None
    
    def fit(self, X: np.ndarray, y_multi: np.ndarray) -> 'SingleTaskGBDT':
        """
        各タスクに対して独立したモデルを学習
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクのターゲット値
            
        Returns:
        --------
        self : SingleTaskGBDT
            学習済みモデル
        """
        self.n_tasks = y_multi.shape[1]
        self.trees = [[] for _ in range(self.n_tasks)]
        self.initial_predictions = []
        
        # 各タスクに対して独立したモデルを学習
        for task_idx in range(self.n_tasks):
            # 現在のタスクのターゲット値
            y_task = y_multi[:, task_idx]
            
            # 初期予測値を計算（平均値）
            initial_pred = np.mean(y_task)
            self.initial_predictions.append(initial_pred)
            
            # 現在の予測値を初期化
            y_pred = np.full(X.shape[0], initial_pred)
            
            # 各反復で木を構築
            for i in range(self.n_estimators):
                # 勾配とヘシアンを計算（二乗誤差損失）
                gradients = 2 * (y_pred - y_task)
                hessians = np.ones_like(gradients) * 2
                
                # 単一タスク用に勾配とヘシアンを2次元配列に変換
                gradients_2d = gradients.reshape(-1, 1)
                hessians_2d = hessians.reshape(-1, 1)
                
                # 新しい木を構築
                tree = MultiTaskDecisionTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state
                )
                
                # 木を学習
                tree.fit(
                    X, 
                    gradients_2d, 
                    hessians_2d
                )
                
                # 木を保存
                self.trees[task_idx].append(tree)
                
                # 予測値を更新
                update = tree.predict(X)[:, 0]  # 単一タスクなので最初の列のみ使用
                y_pred += self.learning_rate * update
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        学習済みモデルで予測
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
            
        Returns:
        --------
        y_pred : array-like, shape=(n_samples, n_tasks)
            各タスクの予測値
        """
        if not self.trees:
            raise ValueError("Model has not been trained yet")
        
        # 各タスクの予測値を計算
        y_pred = np.zeros((X.shape[0], self.n_tasks))
        
        for task_idx in range(self.n_tasks):
            # 初期予測値
            y_pred[:, task_idx] = self.initial_predictions[task_idx]
            
            # 各木の予測を加算
            for tree in self.trees[task_idx]:
                y_pred[:, task_idx] += self.learning_rate * tree.predict(X)[:, 0]
        
        return y_pred
    
    def evaluate(self, X: np.ndarray, y_multi: np.ndarray, metrics: List[str] = ['mse']) -> Dict:
        """
        モデルの評価
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            入力特徴量
        y_multi : array-like, shape=(n_samples, n_tasks)
            複数タスクの真のターゲット値
        metrics : list of str, default=['mse']
            使用する評価指標のリスト
            
        Returns:
        --------
        results : dict
            各タスクの各評価指標の値
        """
        # 予測
        y_pred = self.predict(X)
        
        # 結果格納用辞書
        results = {}
        
        # 各評価指標を計算
        for metric in metrics:
            if metric.lower() == 'mse':
                # 平均二乗誤差
                task_mse = np.mean((y_multi - y_pred) ** 2, axis=0)
                results['mse'] = task_mse
                results['mse_avg'] = np.mean(task_mse)
            
            elif metric.lower() == 'rmse':
                # 平方根平均二乗誤差
                task_rmse = np.sqrt(np.mean((y_multi - y_pred) ** 2, axis=0))
                results['rmse'] = task_rmse
                results['rmse_avg'] = np.mean(task_rmse)
            
            elif metric.lower() == 'mae':
                # 平均絶対誤差
                task_mae = np.mean(np.abs(y_multi - y_pred), axis=0)
                results['mae'] = task_mae
                results['mae_avg'] = np.mean(task_mae)
            
            elif metric.lower() == 'r2':
                # 決定係数
                y_mean = np.mean(y_multi, axis=0)
                ss_tot = np.sum((y_multi - y_mean.reshape(1, -1)) ** 2, axis=0)
                ss_res = np.sum((y_multi - y_pred) ** 2, axis=0)
                task_r2 = 1 - (ss_res / ss_tot)
                results['r2'] = task_r2
                results['r2_avg'] = np.mean(task_r2)
            
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        return results


def run_mt_vs_st_comparison(dataset_name: str,
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
    マルチタスク学習と単一タスク学習を比較
    
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
    
    # マルチタスクモデルと単一タスクモデルを評価
    models = {
        'MT-GBDT': MTGBDT(**common_params),
        'ST-GBDT': SingleTaskGBDT(**common_params),
        'MT-XGBoost': MTXGBoost(**common_params),
        'ST-XGBoost': SingleTaskXGBoost(**common_params),
        'MT-LightGBM': MTLightGBM(**common_params),
        'ST-LightGBM': SingleTaskLightGBM(**common_params)
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
    with open(os.path.join(output_dir, f"{dataset_name}_mt_vs_st.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 結果をプロット
    plot_mt_vs_st_results(results, os.path.join(output_dir, f"{dataset_name}_mt_vs_st.png"))
    
    return results


def plot_mt_vs_st_results(results: Dict, save_path: Optional[str] = None) -> None:
    """
    マルチタスク学習と単一タスク学習の比較結果をプロット
    
    Parameters:
    -----------
    results : dict
        比較結果
    save_path : str, optional
        保存先のパス
    """
    # モデル名
    model_names = list(results['models'].keys())
    
    # マルチタスクモデルと単一タスクモデルを分類
    mt_models = [name for name in model_names if name.startswith('MT-')]
    st_models = [name for name in model_names if name.startswith('ST-')]
    
    # 各モデルタイプごとに色を設定
    colors = {
        'GBDT': 'blue',
        'XGBoost': 'green',
        'LightGBM': 'red'
    }
    
    # MSE
    mse_values = [results['models'][model]['evaluation']['mse_avg'] for model in model_names]
    
    # R2
    r2_values = [results['models'][model]['evaluation']['r2_avg'] for model in model_names]
    
    # 訓練時間
    train_times = [results['models'][model]['train_time'] for model in model_names]
    
    # 予測時間
    predict_times = [results['models'][model]['predict_time'] for model in model_names]
    
    # プロット
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # タイトル
    fig.suptitle(f"Multi-Task vs Single-Task Learning on {results['dataset']} Dataset", fontsize=16)
    
    # MSE
    bar_positions = np.arange(len(model_names))
    bars = axes[0, 0].bar(bar_positions, mse_values, color=[colors[model.split('-')[1]] for model in model_names])
    axes[0, 0].set_title('Mean Squared Error (MSE)')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_xticks(bar_positions)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    
    # MSEの値をバーの上に表示
    for bar, value in zip(bars, mse_values):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                      f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    # R2
    bars = axes[0, 1].bar(bar_positions, r2_values, color=[colors[model.split('-')[1]] for model in model_names])
    axes[0, 1].set_title('R² Score')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].set_xticks(bar_positions)
    axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
    
    # R2の値をバーの上に表示
    for bar, value in zip(bars, r2_values):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                      f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 訓練時間
    bars = axes[1, 0].bar(bar_positions, train_times, color=[colors[model.split('-')[1]] for model in model_names])
    axes[1, 0].set_title('Training Time (s)')
    axes[1, 0].set_ylabel('Time (s)')
    axes[1, 0].set_xticks(bar_positions)
    axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
    
    # 訓練時間の値をバーの上に表示
    for bar, value in zip(bars, train_times):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                      f'{value:.2f}s', ha='center', va='bottom', fontsize=8)
    
    # 予測時間
    bars = axes[1, 1].bar(bar_positions, predict_times, color=[colors[model.split('-')[1]] for model in model_names])
    axes[1, 1].set_title('Prediction Time (s)')
    axes[1, 1].set_ylabel('Time (s)')
    axes[1, 1].set_xticks(bar_positions)
    axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
    
    # 予測時間の値をバーの上に表示
    for bar, value in zip(bars, predict_times):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                      f'{value:.2f}s', ha='center', va='bottom', fontsize=8)
    
    # 凡例を追加
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=model_type) for model_type, color in colors.items()]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.98))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_task_specific_mt_vs_st(results: Dict, save_path: Optional[str] = None) -> None:
    """
    タスク別のマルチタスク学習と単一タスク学習の比較結果をプロット
    
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
    fig, axes = plt.subplots(n_tasks, 2, figsize=(16, 6 * n_tasks))
    
    # タイトル
    fig.suptitle(f"Task-Specific Performance on {results['dataset']} Dataset", fontsize=16)
    
    # 各モデルタイプごとに色を設定
    colors = {
        'GBDT': 'blue',
        'XGBoost': 'green',
        'LightGBM': 'red'
    }
    
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
        
        bar_positions = np.arange(len(model_names))
        bars = ax_mse.bar(bar_positions, mse_values, color=[colors[model.split('-')[1]] for model in model_names])
        ax_mse.set_title(f'Task {task_idx+1} - Mean Squared Error (MSE)')
        ax_mse.set_ylabel('MSE')
        ax_mse.set_xticks(bar_positions)
        ax_mse.set_xticklabels(model_names, rotation=45, ha='right')
        
        # MSEの値をバーの上に表示
        for bar, value in zip(bars, mse_values):
            ax_mse.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        # R2プロット
        bars = ax_r2.bar(bar_positions, r2_values, color=[colors[model.split('-')[1]] for model in model_names])
        ax_r2.set_title(f'Task {task_idx+1} - R² Score')
        ax_r2.set_ylabel('R²')
        ax_r2.set_xticks(bar_positions)
        ax_r2.set_xticklabels(model_names, rotation=45, ha='right')
        
        # R2の値をバーの上に表示
        for bar, value in zip(bars, r2_values):
            ax_r2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                      f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 凡例を追加
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=model_type) for model_type, color in colors.items()]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 0.98))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def run_all_mt_vs_st_experiments(output_dir: str = "results", random_state: int = 42) -> None:
    """
    すべてのマルチタスク学習と単一タスク学習の比較実験を実行
    
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
        print(f"Running MT vs ST experiment: {exp['name']}")
        print(f"{'='*50}")
        
        # データを生成/読み込み
        X_train, y_train, X_test, y_test = exp['data_func'](**exp['params'])
        
        # モデル比較を実行
        results = run_mt_vs_st_comparison(
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
        plot_task_specific_mt_vs_st(
            results,
            save_path=os.path.join(output_dir, f"{exp['name']}_mt_vs_st_task_specific.png")
        )
        
        all_results[exp['name']] = results
    
    # すべての実験の要約を作成
    summary = {
        'datasets': [],
        'mt_vs_st_mse_ratio': {},
        'mt_vs_st_r2_ratio': {},
        'mt_vs_st_train_time_ratio': {}
    }
    
    for dataset_name, result in all_results.items():
        summary['datasets'].append(dataset_name)
        
        for model_type in ['GBDT', 'XGBoost', 'LightGBM']:
            mt_model = f'MT-{model_type}'
            st_model = f'ST-{model_type}'
            
            if mt_model not in result['models'] or st_model not in result['models']:
                continue
            
            # MSE比率（ST/MT、1より大きいとMTが優れている）
            mse_ratio = result['models'][st_model]['evaluation']['mse_avg'] / result['models'][mt_model]['evaluation']['mse_avg']
            
            # R2比率（MT/ST、1より大きいとMTが優れている）
            # R2は負の値になる可能性があるため、単純な比率ではなく差分を使用
            r2_diff = result['models'][mt_model]['evaluation']['r2_avg'] - result['models'][st_model]['evaluation']['r2_avg']
            
            # 訓練時間比率（ST/MT、1より小さいとSTが速い）
            train_time_ratio = result['models'][mt_model]['evaluation']['mse_avg'] / result['models'][st_model]['evaluation']['mse_avg']
            
            if model_type not in summary['mt_vs_st_mse_ratio']:
                summary['mt_vs_st_mse_ratio'][model_type] = []
                summary['mt_vs_st_r2_ratio'][model_type] = []
                summary['mt_vs_st_train_time_ratio'][model_type] = []
            
            summary['mt_vs_st_mse_ratio'][model_type].append(mse_ratio)
            summary['mt_vs_st_r2_ratio'][model_type].append(r2_diff)
            summary['mt_vs_st_train_time_ratio'][model_type].append(train_time_ratio)
    
    # 要約をJSONファイルに保存
    with open(os.path.join(output_dir, "mt_vs_st_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 要約をプロット
    plot_mt_vs_st_summary(summary, os.path.join(output_dir, "mt_vs_st_summary.png"))


def plot_mt_vs_st_summary(summary: Dict, save_path: Optional[str] = None) -> None:
    """
    マルチタスク学習と単一タスク学習の比較要約をプロット
    
    Parameters:
    -----------
    summary : dict
        比較要約
    save_path : str, optional
        保存先のパス
    """
    # モデルタイプ
    model_types = list(summary['mt_vs_st_mse_ratio'].keys())
    
    # データセット名
    dataset_names = summary['datasets']
    
    # プロット
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # タイトル
    fig.suptitle("Multi-Task vs Single-Task Learning Performance Summary", fontsize=16)
    
    # MSE比率
    mse_df = pd.DataFrame({
        model_type: summary['mt_vs_st_mse_ratio'][model_type] for model_type in model_types
    }, index=dataset_names)
    
    sns.heatmap(mse_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[0])
    axes[0].set_title('MSE Ratio (ST/MT)\nHigher is better for MT')
    
    # R2差分
    r2_df = pd.DataFrame({
        model_type: summary['mt_vs_st_r2_ratio'][model_type] for model_type in model_types
    }, index=dataset_names)
    
    sns.heatmap(r2_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[1])
    axes[1].set_title('R² Difference (MT-ST)\nHigher is better for MT')
    
    # 訓練時間比率
    time_df = pd.DataFrame({
        model_type: summary['mt_vs_st_train_time_ratio'][model_type] for model_type in model_types
    }, index=dataset_names)
    
    sns.heatmap(time_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[2])
    axes[2].set_title('MSE Ratio (MT/ST)\nHigher is better for MT')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


if __name__ == "__main__":
    # すべての実験を実行
    run_all_mt_vs_st_experiments(output_dir="results", random_state=42)
