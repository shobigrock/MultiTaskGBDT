"""
MT-GBM実装のテストと動作確認用スクリプト

このスクリプトは、MT-GBM実装の各コンポーネントをテストし、
基本的な動作確認を行います。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 各モジュールをインポート
from mt_gbm.models.base import MTGBMBase
from mt_gbm.models.gbdt import MTGBDT
from mt_gbm.models.xgboost_wrapper import MTXGBoost
from mt_gbm.models.lgbm_wrapper import MTLightGBM
from mt_gbm.data.synthetic import generate_synthetic_data, generate_nonlinear_synthetic_data
from mt_gbm.data.real_data import load_california_housing
from mt_gbm.utils.model_interface import test_model_interface, compare_models


def test_data_generation():
    """データ生成モジュールのテスト"""
    print("\n=== テスト: データ生成モジュール ===")
    
    # 線形データ生成
    print("線形データ生成テスト...")
    X_train, y_train, X_test, y_test = generate_synthetic_data(
        n_samples=100, n_features=5, n_tasks=2, random_state=42
    )
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # 非線形データ生成
    print("\n非線形データ生成テスト...")
    X_train, y_train, X_test, y_test = generate_nonlinear_synthetic_data(
        n_samples=100, n_features=5, n_tasks=2, random_state=42
    )
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # 実データ読み込み
    print("\n実データ読み込みテスト...")
    try:
        X_train, y_train, X_test, y_test = load_california_housing(random_state=42)
        print(f"California Housing データセット読み込み成功")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
    except Exception as e:
        print(f"California Housing データセット読み込みエラー: {e}")
    
    print("データ生成モジュールのテスト完了")


def test_model_implementations():
    """モデル実装のテスト"""
    print("\n=== テスト: モデル実装 ===")
    
    # テスト用データ生成
    print("テストデータ生成...")
    X_train, y_train, X_test, y_test = generate_synthetic_data(
        n_samples=100, n_features=5, n_tasks=2, random_state=42
    )
    
    # MTGBDT
    print("\nMTGBDTテスト...")
    try:
        model = MTGBDT(n_estimators=10, learning_rate=0.1, max_depth=3, random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        
        eval_results = model.evaluate(X_test, y_test, metrics=['mse', 'rmse', 'r2'])
        
        print(f"訓練時間: {train_time:.4f}秒")
        print(f"予測時間: {predict_time:.4f}秒")
        print(f"MSE: {eval_results['mse_avg']:.6f}")
        print(f"RMSE: {eval_results['rmse_avg']:.6f}")
        print(f"R2: {eval_results['r2_avg']:.6f}")
        print("MTGBDTテスト成功")
    except Exception as e:
        print(f"MTGBDTテストエラー: {e}")
    
    # MTXGBoost
    print("\nMTXGBoostテスト...")
    try:
        model = MTXGBoost(n_estimators=10, learning_rate=0.1, max_depth=3, random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        
        eval_results = model.evaluate(X_test, y_test, metrics=['mse', 'rmse', 'r2'])
        
        print(f"訓練時間: {train_time:.4f}秒")
        print(f"予測時間: {predict_time:.4f}秒")
        print(f"MSE: {eval_results['mse_avg']:.6f}")
        print(f"RMSE: {eval_results['rmse_avg']:.6f}")
        print(f"R2: {eval_results['r2_avg']:.6f}")
        print("MTXGBoostテスト成功")
    except Exception as e:
        print(f"MTXGBoostテストエラー: {e}")
    
    # MTLightGBM
    print("\nMTLightGBMテスト...")
    try:
        model = MTLightGBM(n_estimators=10, learning_rate=0.1, max_depth=3, random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time
        
        eval_results = model.evaluate(X_test, y_test, metrics=['mse', 'rmse', 'r2'])
        
        print(f"訓練時間: {train_time:.4f}秒")
        print(f"予測時間: {predict_time:.4f}秒")
        print(f"MSE: {eval_results['mse_avg']:.6f}")
        print(f"RMSE: {eval_results['rmse_avg']:.6f}")
        print(f"R2: {eval_results['r2_avg']:.6f}")
        print("MTLightGBMテスト成功")
    except Exception as e:
        print(f"MTLightGBMテストエラー: {e}")
    
    print("モデル実装のテスト完了")


def test_experiment_scripts():
    """実験スクリプトのテスト"""
    print("\n=== テスト: 実験スクリプト ===")
    
    # モデルインターフェーステスト
    print("\nモデルインターフェーステスト...")
    try:
        # テスト用データ生成
        X_train, y_train, X_test, y_test = generate_synthetic_data(
            n_samples=100, n_features=5, n_tasks=2, random_state=42
        )
        
        # MTGBDTのテスト
        results = test_model_interface(
            MTGBDT,
            model_params={'n_estimators': 10, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42},
            n_samples=100,
            n_features=5,
            n_tasks=2,
            random_state=42
        )
        
        print(f"MTGBDT MSE: {results['evaluation']['mse_avg']:.6f}")
        print(f"MTGBDT R2: {results['evaluation']['r2_avg']:.6f}")
        print("モデルインターフェーステスト成功")
    except Exception as e:
        print(f"モデルインターフェーステストエラー: {e}")
    
    # 比較実験テスト
    print("\n比較実験テスト...")
    try:
        # 小規模データで比較実験
        results = compare_models(
            n_samples=100,
            n_features=5,
            n_tasks=2,
            random_state=42
        )
        
        # 結果表示
        for model_name, model_results in results.items():
            print(f"{model_name} MSE: {model_results['evaluation']['mse_avg']:.6f}")
            print(f"{model_name} R2: {model_results['evaluation']['r2_avg']:.6f}")
            print(f"{model_name} 訓練時間: {model_results['train_time']:.4f}秒")
        
        print("比較実験テスト成功")
    except Exception as e:
        print(f"比較実験テストエラー: {e}")
    
    print("実験スクリプトのテスト完了")


def test_directory_structure():
    """ディレクトリ構造のテスト"""
    print("\n=== テスト: ディレクトリ構造 ===")
    
    # プロジェクトルートディレクトリ
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 必要なディレクトリとファイルのリスト
    required_dirs = [
        'data',
        'models',
        'experiments',
        'utils'
    ]
    
    required_files = [
        'README.md',
        'requirements.txt',
        'models/__init__.py',
        'models/base.py',
        'models/gbdt.py',
        'models/xgboost_wrapper.py',
        'models/lgbm_wrapper.py',
        'data/__init__.py',
        'data/synthetic.py',
        'data/real_data.py',
        'experiments/__init__.py',
        'experiments/compare_models.py',
        'experiments/compare_mt_st.py',
        'utils/__init__.py',
        'utils/model_interface.py',
        'utils/visualization.py'
    ]
    
    # ディレクトリのチェック
    for dir_name in required_dirs:
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            print(f"✓ ディレクトリ存在: {dir_name}")
        else:
            print(f"✗ ディレクトリ欠落: {dir_name}")
    
    # ファイルのチェック
    for file_name in required_files:
        file_path = os.path.join(root_dir, file_name)
        if os.path.isfile(file_path):
            print(f"✓ ファイル存在: {file_name}")
        else:
            print(f"✗ ファイル欠落: {file_name}")
    
    print("ディレクトリ構造のテスト完了")


def create_test_results_directory():
    """テスト結果ディレクトリの作成"""
    # プロジェクトルートディレクトリ
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # テスト結果ディレクトリ
    test_results_dir = os.path.join(root_dir, 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)
    
    # サブディレクトリ
    os.makedirs(os.path.join(test_results_dir, 'model_comparison'), exist_ok=True)
    os.makedirs(os.path.join(test_results_dir, 'mt_vs_st'), exist_ok=True)
    os.makedirs(os.path.join(test_results_dir, 'figures'), exist_ok=True)
    
    return test_results_dir


def run_all_tests():
    """すべてのテストを実行"""
    print("=== MT-GBM実装テスト開始 ===")
    
    # テスト結果ディレクトリの作成
    test_results_dir = create_test_results_directory()
    print(f"テスト結果ディレクトリ: {test_results_dir}")
    
    # ディレクトリ構造のテスト
    test_directory_structure()
    
    # データ生成モジュールのテスト
    test_data_generation()
    
    # モデル実装のテスト
    test_model_implementations()
    
    # 実験スクリプトのテスト
    test_experiment_scripts()
    
    print("\n=== MT-GBM実装テスト完了 ===")
    print(f"すべてのテストが完了しました。テスト結果は {test_results_dir} に保存されています。")


if __name__ == "__main__":
    run_all_tests()
