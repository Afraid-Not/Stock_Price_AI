"""
Optuna 하이퍼파라미터 튜닝
- XGBoost + LightGBM + CatBoost 최적 파라미터 탐색
"""
import pandas as pd
import numpy as np
import random
import warnings
import argparse
import optuna
from optuna.samplers import TPESampler
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import os
from datetime import datetime

# 9개 종목 코드
TARGET_STOCKS = [
    '005930', '000660', '035420', '035720', '006400',
    '066570', '034220', '018260', '030200'
]


def set_seed(seed: int):
    """모든 랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class OptunaOptimizer:
    def __init__(self, data_path, n_splits=5, lag_days=[1, 3, 5],
                 target_threshold=0.01, seed=42, n_trials=50):
        self.data_path = data_path
        self.n_splits = n_splits
        self.lag_days = lag_days
        self.target_threshold = target_threshold
        self.seed = seed
        self.n_trials = n_trials
        self.label_encoder = LabelEncoder()
        
        set_seed(seed)
        
        # 데이터 로드 및 전처리
        self.X, self.y = self.prepare_data()
        
    def prepare_data(self):
        """데이터 준비"""
        print("📂 데이터 로드 중...")
        df = pd.read_csv(self.data_path)
        
        # stock_code 문자열 변환
        df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
        
        # 9개 종목 필터
        df = df[df['stock_code'].isin(TARGET_STOCKS)].copy()
        
        # 날짜 처리 - ⚠️ TimeSeriesSplit을 위해 날짜 우선 정렬!
        df['날짜'] = pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['날짜'])
        df = df.sort_values(['날짜', 'stock_code']).reset_index(drop=True)
        
        print(f"   원본: {len(df):,}건")
        
        # 타겟 재정의
        if 'next_rtn' in df.columns:
            df = df.dropna(subset=['next_rtn'])
            df_up = df[df['next_rtn'] >= self.target_threshold].copy()
            df_up['target'] = 1
            df_down = df[df['next_rtn'] <= -self.target_threshold].copy()
            df_down['target'] = 0
            df = pd.concat([df_up, df_down], ignore_index=True)
            df = df.sort_values(['날짜', 'stock_code']).reset_index(drop=True)
            df = df.drop(columns=['next_rtn'])
        
        print(f"   필터링 후: {len(df):,}건")
        
        # Lag 피처
        df = self.add_lag_features(df)
        print(f"   Lag 추가 후: {len(df):,}건")
        
        # 피처 추출
        exclude_cols = ['날짜', 'target', 'stock_code', 'stock_code_encoded',
                        '시가', '고가', '저가', '종가', '거래량', '거래대금',
                        'stock_name', 'next_rtn']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # stock_code 인코딩
        df['stock_code_encoded'] = self.label_encoder.fit_transform(df['stock_code'])
        feature_cols.append('stock_code_encoded')
        
        X = df[feature_cols].values
        y = df['target'].values
        
        print(f"   피처 수: {len(feature_cols)}")
        print(f"   클래스: 0={sum(y==0):,}, 1={sum(y==1):,}")
        
        return X, y
    
    def add_lag_features(self, df):
        """Lag 피처 추가"""
        base_features = [
            'open_gap', 'high_ratio', 'low_ratio', 'volatility',
            '개인_체결강도', '외국인_체결강도', '기관계_체결강도',
            'vol_ratio', 'rsi'
        ]
        base_features = [f for f in base_features if f in df.columns]
        
        lag_dfs = []
        for stock_code in df['stock_code'].unique():
            stock_df = df[df['stock_code'] == stock_code].copy()
            stock_df = stock_df.sort_values('날짜')
            
            for lag in self.lag_days:
                for feat in base_features:
                    stock_df[f'{feat}_lag{lag}'] = stock_df[feat].shift(lag)
            
            lag_dfs.append(stock_df)
        
        df = pd.concat(lag_dfs, ignore_index=True)
        df = df.dropna()
        # ⚠️ TimeSeriesSplit을 위해 날짜 우선 정렬!
        df = df.sort_values(['날짜', 'stock_code']).reset_index(drop=True)
        return df
    
    def objective_xgb(self, trial):
        """XGBoost 목적 함수"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': self.seed,
            'eval_metric': 'auc',
            'early_stopping_rounds': 50
        }
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred)
            scores.append(f1)
        
        return np.mean(scores)
    
    def objective_lgb(self, trial):
        """LightGBM 목적 함수"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'random_state': self.seed,
            'verbose': -1
        }
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred)
            scores.append(f1)
        
        return np.mean(scores)
    
    def objective_cat(self, trial):
        """CatBoost 목적 함수"""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'random_state': self.seed,
            'eval_metric': 'AUC',
            'early_stopping_rounds': 50,
            'verbose': False
        }
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(self.X):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            model = CatBoostClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val)
            )
            
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred)
            scores.append(f1)
        
        return np.mean(scores)
    
    def run(self):
        """최적화 실행"""
        print("\n" + "=" * 60)
        print("🔬 Optuna 하이퍼파라미터 튜닝")
        print("=" * 60)
        print(f"시드: {self.seed}")
        print(f"시행 횟수: {self.n_trials}")
        print(f"CV Folds: {self.n_splits}")
        
        results = {}
        
        # XGBoost 최적화
        print("\n" + "-" * 40)
        print("🔵 XGBoost 튜닝 중...")
        print("-" * 40)
        
        study_xgb = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.seed)
        )
        study_xgb.optimize(
            self.objective_xgb, 
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        results['xgboost'] = {
            'best_params': study_xgb.best_params,
            'best_f1': study_xgb.best_value
        }
        
        print(f"\n✅ XGBoost 최적 F1: {study_xgb.best_value:.4f}")
        print("   최적 파라미터:")
        for k, v in study_xgb.best_params.items():
            print(f"      {k}: {v}")
        
        # LightGBM 최적화
        print("\n" + "-" * 40)
        print("🟢 LightGBM 튜닝 중...")
        print("-" * 40)
        
        study_lgb = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.seed)
        )
        study_lgb.optimize(
            self.objective_lgb, 
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        results['lightgbm'] = {
            'best_params': study_lgb.best_params,
            'best_f1': study_lgb.best_value
        }
        
        print(f"\n✅ LightGBM 최적 F1: {study_lgb.best_value:.4f}")
        print("   최적 파라미터:")
        for k, v in study_lgb.best_params.items():
            print(f"      {k}: {v}")
        
        # CatBoost 최적화
        print("\n" + "-" * 40)
        print("🟡 CatBoost 튜닝 중...")
        print("-" * 40)
        
        study_cat = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.seed)
        )
        study_cat.optimize(
            self.objective_cat,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        results['catboost'] = {
            'best_params': study_cat.best_params,
            'best_f1': study_cat.best_value
        }
        
        print(f"\n✅ CatBoost 최적 F1: {study_cat.best_value:.4f}")
        print("   최적 파라미터:")
        for k, v in study_cat.best_params.items():
            print(f"      {k}: {v}")
        
        # 결과 저장
        os.makedirs('tuning_results', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result_path = f"tuning_results/optuna_{timestamp}.pkl"
        joblib.dump(results, result_path)
        print(f"\n💾 결과 저장: {result_path}")
        
        # 텍스트로도 저장
        txt_path = f"tuning_results/best_params_{timestamp}.txt"
        with open(txt_path, 'w') as f:
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Trials: {self.n_trials}\n")
            f.write(f"Threshold: {self.target_threshold}\n\n")
            
            f.write("=" * 40 + "\n")
            f.write("XGBoost\n")
            f.write("=" * 40 + "\n")
            f.write(f"Best F1: {results['xgboost']['best_f1']:.4f}\n")
            for k, v in results['xgboost']['best_params'].items():
                f.write(f"{k}: {v}\n")
            
            f.write("\n" + "=" * 40 + "\n")
            f.write("LightGBM\n")
            f.write("=" * 40 + "\n")
            f.write(f"Best F1: {results['lightgbm']['best_f1']:.4f}\n")
            for k, v in results['lightgbm']['best_params'].items():
                f.write(f"{k}: {v}\n")
            
            f.write("\n" + "=" * 40 + "\n")
            f.write("CatBoost\n")
            f.write("=" * 40 + "\n")
            f.write(f"Best F1: {results['catboost']['best_f1']:.4f}\n")
            for k, v in results['catboost']['best_params'].items():
                f.write(f"{k}: {v}\n")
        
        print(f"💾 파라미터 저장: {txt_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Optuna 하이퍼파라미터 튜닝")
    parser.add_argument("--data", type=str, default="_data/merged_with_macro.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_trials", type=int, default=50, help="시행 횟수")
    parser.add_argument("--n_splits", type=int, default=5, help="CV Fold 수")
    parser.add_argument("--threshold", type=float, default=0.01)
    args = parser.parse_args()
    
    optimizer = OptunaOptimizer(
        data_path=args.data,
        n_splits=args.n_splits,
        target_threshold=args.threshold,
        seed=args.seed,
        n_trials=args.n_trials
    )
    
    results = optimizer.run()
    
    print("\n" + "=" * 60)
    print("🎉 튜닝 완료!")
    print("=" * 60)
    print(f"\n📊 최종 결과:")
    print(f"   XGBoost F1: {results['xgboost']['best_f1']:.4f}")
    print(f"   LightGBM F1: {results['lightgbm']['best_f1']:.4f}")
    print(f"   CatBoost F1: {results['catboost']['best_f1']:.4f}")


if __name__ == "__main__":
    main()

