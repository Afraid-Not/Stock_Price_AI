"""
CatBoost 전용 학습 + Optuna 튜닝
- 9개 종목: 삼성전자, SK하이닉스, NAVER, 카카오, 삼성SDI, LG전자, LG디스플레이, 삼성SDS, KT
- 타겟: 오늘 종가 → 내일 종가 (1% 이상 상승/하락)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import joblib
import os
from datetime import datetime


def set_seed(seed: int):
    """모든 랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# 9개 종목 코드
TARGET_STOCKS = [
    '005930',  # 삼성전자
    '000660',  # SK하이닉스
    '035420',  # NAVER
    '035720',  # 카카오
    '006400',  # 삼성SDI
    '066570',  # LG전자
    '034220',  # LG디스플레이
    '018260',  # 삼성SDS
    '030200',  # KT
]

STOCK_NAMES = {
    '005930': '삼성전자',
    '000660': 'SK하이닉스',
    '035420': 'NAVER',
    '035720': '카카오',
    '006400': '삼성SDI',
    '066570': 'LG전자',
    '034220': 'LG디스플레이',
    '018260': '삼성SDS',
    '030200': 'KT',
}


class CatBoostTrainer:
    def __init__(self, data_path, n_splits=5, lag_days=[1, 3, 5], 
                 target_threshold=0.01, seed=42):
        self.data_path = data_path
        self.n_splits = n_splits
        self.lag_days = lag_days
        self.target_threshold = target_threshold
        self.seed = seed
        self.model = None
        self.label_encoder = LabelEncoder()
        self.best_params = None
        
        set_seed(seed)
        
        self.model_dir = 'models_catboost'
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs('tuning_results', exist_ok=True)
        
    def load_data(self):
        """9개 종목만 로드"""
        print("📂 데이터 로드 중...")
        df = pd.read_csv(self.data_path)
        
        # stock_code를 문자열로 변환
        df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
        
        # 9개 종목만 필터링
        df = df[df['stock_code'].isin(TARGET_STOCKS)].copy()
        
        # 날짜 정렬 - TimeSeriesSplit을 위해 날짜 우선!
        df['날짜'] = pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['날짜'])
        df = df.sort_values(['날짜', 'stock_code']).reset_index(drop=True)
        
        print(f"   원본 데이터: {len(df):,}건")
        print(f"   기간: {df['날짜'].min().date()} ~ {df['날짜'].max().date()}")
        print(f"   종목 수: {df['stock_code'].nunique()}개")
        
        # 타겟 재정의
        if self.target_threshold is not None:
            df = self.redefine_target(df)
        
        # Lag 피처 추가
        df = self.add_lag_features(df)
        
        print(f"   Lag 피처 추가 후: {len(df):,}건")
        print(f"   클래스 분포: 0={len(df[df['target']==0]):,}, 1={len(df[df['target']==1]):,}")
        
        return df
    
    def redefine_target(self, df):
        """타겟 재정의: 오늘 종가 → 내일 종가"""
        print(f"\n🎯 타겟 재정의 (임계값: ±{self.target_threshold*100:.1f}%)")
        
        if 'next_rtn' not in df.columns:
            raise ValueError("next_rtn 컬럼이 없습니다.")
        
        df = df.dropna(subset=['next_rtn'])
        before_filter = len(df)
        
        df_up = df[df['next_rtn'] >= self.target_threshold].copy()
        df_up['target'] = 1
        
        df_down = df[df['next_rtn'] <= -self.target_threshold].copy()
        df_down['target'] = 0
        
        df_filtered = pd.concat([df_up, df_down], ignore_index=True)
        df_filtered = df_filtered.sort_values(['날짜', 'stock_code']).reset_index(drop=True)
        df_filtered = df_filtered.drop(columns=['next_rtn'])
        
        excluded = before_filter - len(df_filtered)
        print(f"   상승: {len(df_up):,}건, 하락: {len(df_down):,}건")
        print(f"   제외 (노이즈): {excluded:,}건 ({excluded/before_filter*100:.1f}%)")
        
        return df_filtered
    
    def add_lag_features(self, df):
        """Lag 피처 추가"""
        print(f"\n📊 Lag 피처 생성 중... (lag_days: {self.lag_days})")
        
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
        
        df_with_lag = pd.concat(lag_dfs, ignore_index=True)
        
        before_drop = len(df_with_lag)
        df_with_lag = df_with_lag.dropna()
        
        # 날짜 우선 정렬
        df_with_lag = df_with_lag.sort_values(['날짜', 'stock_code']).reset_index(drop=True)
        
        new_features = [c for c in df_with_lag.columns if '_lag' in c]
        print(f"   생성된 Lag 피처 수: {len(new_features)}개")
        print(f"   NaN 제거: {before_drop - len(df_with_lag):,}건")
        
        return df_with_lag
    
    def get_features(self, df):
        """피처 컬럼 추출"""
        exclude_cols = ['날짜', 'target', 'stock_code', 'stock_code_encoded',
                        '시가', '고가', '저가', '종가', '거래량', '거래대금',
                        'stock_name', 'next_rtn']
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        return feature_cols
    
    def prepare_data(self):
        """데이터 준비"""
        df = self.load_data()
        
        feature_cols = self.get_features(df)
        print(f"\n📊 피처 수: {len(feature_cols)}개")
        
        # stock_code 인코딩
        df['stock_code_encoded'] = self.label_encoder.fit_transform(df['stock_code'])
        if 'stock_code_encoded' not in feature_cols:
            feature_cols.append('stock_code_encoded')
        
        X = df[feature_cols].values
        y = df['target'].values
        
        return X, y, feature_cols
    
    def tune(self, n_trials=50, metric='f1'):
        """Optuna 하이퍼파라미터 튜닝"""
        print("\n" + "=" * 60)
        print("🔬 CatBoost Optuna 튜닝")
        print("=" * 60)
        print(f"시드: {self.seed}")
        print(f"시행 횟수: {n_trials}")
        print(f"CV Folds: {self.n_splits}")
        print(f"평가 지표: {metric.upper()}")
        
        X, y, _ = self.prepare_data()
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_state': self.seed,
                'eval_metric': 'AUC',
                'early_stopping_rounds': 50,
                'verbose': False
            }
            
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = CatBoostClassifier(**params)
                model.fit(X_train, y_train, eval_set=(X_val, y_val))
                
                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1]
                
                if metric == 'f1':
                    score = f1_score(y_val, y_pred)
                elif metric == 'auc':
                    score = roc_auc_score(y_val, y_prob)
                else:
                    score = accuracy_score(y_val, y_pred)
                
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        
        print(f"\n✅ 최적 {metric.upper()}: {study.best_value:.4f}")
        print("   최적 파라미터:")
        for k, v in study.best_params.items():
            print(f"      {k}: {v}")
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result = {
            'best_params': study.best_params,
            f'best_{metric}': study.best_value,
            'metric': metric
        }
        
        result_path = f"tuning_results/catboost_{timestamp}.pkl"
        joblib.dump(result, result_path)
        print(f"\n💾 결과 저장: {result_path}")
        
        # 텍스트로도 저장
        txt_path = f"tuning_results/catboost_params_{timestamp}.txt"
        with open(txt_path, 'w') as f:
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Trials: {n_trials}\n")
            f.write(f"Metric: {metric}\n")
            f.write(f"Best {metric}: {study.best_value:.4f}\n\n")
            for k, v in study.best_params.items():
                f.write(f"{k}: {v}\n")
        print(f"💾 파라미터 저장: {txt_path}")
        
        return study.best_params
    
    def train(self, params=None):
        """학습 실행"""
        print("\n" + "=" * 60)
        print("🚀 CatBoost 학습 시작")
        print("=" * 60)
        
        X, y, feature_cols = self.prepare_data()
        
        # 파라미터 설정
        if params is None:
            params = self.best_params if self.best_params else {}
        
        # 기본값 설정
        default_params = {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3.0,
            'bagging_temperature': 0.5,
            'random_strength': 0.1,
            'random_state': self.seed,
            'eval_metric': 'AUC',
            'early_stopping_rounds': 50,
            'verbose': False
        }
        
        for k, v in default_params.items():
            if k not in params:
                params[k] = v
        
        print("\n📋 사용 파라미터:")
        for k, v in params.items():
            print(f"   {k}: {v}")
        
        # TimeSeriesSplit 학습
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        all_metrics = []
        
        print(f"\n📅 {self.n_splits}-Fold TimeSeriesSplit 학습")
        print("=" * 60)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\n[Fold {fold + 1}/{self.n_splits}]")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"   학습: {len(train_idx):,}건, 검증: {len(val_idx):,}건")
            
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
            
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1': f1_score(y_val, y_pred, zero_division=0),
                'auc': roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.5
            }
            all_metrics.append(metrics)
            
            print(f"   Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
            
            # 마지막 Fold 모델 저장
            if fold == self.n_splits - 1:
                self.model = model
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("📋 교차 검증 결과 (평균 ± 표준편차)")
        print("=" * 60)
        
        summary = {'seed': self.seed}
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            values = [m[metric_name] for m in all_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            summary[f'{metric_name}_mean'] = mean_val
            summary[f'{metric_name}_std'] = std_val
            print(f"   {metric_name:10s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # 결과 CSV에 추가
        summary['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        results_file = f"{self.model_dir}/all_results.csv"
        
        if os.path.exists(results_file):
            df_existing = pd.read_csv(results_file)
            df_new = pd.DataFrame([summary])
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = pd.DataFrame([summary])
        
        df_all.to_csv(results_file, index=False)
        print(f"\n📊 결과 추가: {results_file}")
        
        # 모델 저장
        self.save_model(params)
        
        return all_metrics, summary
    
    def save_model(self, params):
        """모델 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "=" * 60)
        print("💾 모델 저장")
        print("=" * 60)
        
        # 모델 저장
        model_path = f"{self.model_dir}/catboost_{timestamp}.pkl"
        joblib.dump(self.model, model_path)
        print(f"   모델: {model_path}")
        
        # LabelEncoder 저장
        le_path = f"{self.model_dir}/label_encoder_{timestamp}.pkl"
        joblib.dump(self.label_encoder, le_path)
        print(f"   LabelEncoder: {le_path}")
        
        # 파라미터 저장
        params_path = f"{self.model_dir}/params_{timestamp}.pkl"
        joblib.dump(params, params_path)
        print(f"   파라미터: {params_path}")
        
        # 종목 목록 저장
        stocks_path = f"{self.model_dir}/target_stocks_{timestamp}.txt"
        with open(stocks_path, 'w', encoding='utf-8') as f:
            for code in TARGET_STOCKS:
                f.write(f"{code},{STOCK_NAMES[code]}\n")
        print(f"   종목 목록: {stocks_path}")


def main():
    parser = argparse.ArgumentParser(description="CatBoost 전용 학습")
    parser.add_argument("--data", type=str, default="_data/merged_with_macro.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.01)
    
    # 모드 선택
    parser.add_argument("--mode", type=str, choices=['tune', 'train', 'all'], 
                        default='all', help="tune: 튜닝만, train: 학습만, all: 둘 다")
    parser.add_argument("--n_trials", type=int, default=50, help="Optuna 시행 횟수")
    parser.add_argument("--metric", type=str, choices=['f1', 'auc', 'accuracy'],
                        default='f1', help="최적화 지표")
    
    # 직접 파라미터 지정 (train 모드용)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    
    args = parser.parse_args()
    
    print(f"📂 데이터: {args.data}")
    print(f"🎲 시드: {args.seed}")
    print(f"📊 모드: {args.mode}")
    
    trainer = CatBoostTrainer(
        data_path=args.data,
        n_splits=args.n_splits,
        target_threshold=args.threshold,
        seed=args.seed
    )
    
    # 직접 파라미터 지정
    custom_params = {}
    if args.iterations:
        custom_params['iterations'] = args.iterations
    if args.depth:
        custom_params['depth'] = args.depth
    if args.learning_rate:
        custom_params['learning_rate'] = args.learning_rate
    
    if args.mode == 'tune':
        trainer.tune(n_trials=args.n_trials, metric=args.metric)
        
    elif args.mode == 'train':
        trainer.train(params=custom_params if custom_params else None)
        
    elif args.mode == 'all':
        # 튜닝 후 학습
        best_params = trainer.tune(n_trials=args.n_trials, metric=args.metric)
        trainer.train(params=best_params)
    
    print("\n" + "=" * 60)
    print("✅ 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

