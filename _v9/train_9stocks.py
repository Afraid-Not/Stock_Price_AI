"""
9개 종목 전용 앙상블 학습
- 삼성전자, SK하이닉스, NAVER, 카카오, 삼성SDI, LG전자, LG디스플레이, 삼성SDS, KT
- 타겟: 오늘 종가 → 내일 종가 (1% 이상 상승/하락)
"""
import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
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


class Stock9Trainer:
    def __init__(self, data_path, n_splits=5, lag_days=[1, 3, 5], 
                 target_threshold=0.01, seed=42):
        self.data_path = data_path
        self.n_splits = n_splits
        self.lag_days = lag_days
        self.target_threshold = target_threshold
        self.seed = seed
        self.models = {}
        self.label_encoder = LabelEncoder()
        
        # 시드 고정
        set_seed(seed)
        print(f"🎲 랜덤 시드: {seed}")
        
        self.model_dir = 'models_9stocks'
        os.makedirs(self.model_dir, exist_ok=True)
        
    def load_data(self):
        """9개 종목만 로드"""
        print("📂 데이터 로드 중...")
        df = pd.read_csv(self.data_path)
        
        # stock_code를 문자열로 변환 (앞에 0 패딩)
        df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)
        
        # 9개 종목만 필터링
        df = df[df['stock_code'].isin(TARGET_STOCKS)].copy()
        
        # 날짜 정렬 (YYYYMMDD 문자열 형식 처리)
        # ⚠️ TimeSeriesSplit이 제대로 작동하려면 날짜 우선 정렬 필수!
        df['날짜'] = pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['날짜'])  # 파싱 실패한 행 제거
        df = df.sort_values(['날짜', 'stock_code']).reset_index(drop=True)
        
        print(f"   원본 데이터: {len(df):,}건")
        print(f"   기간: {df['날짜'].min().date()} ~ {df['날짜'].max().date()}")
        print(f"   종목 수: {df['stock_code'].nunique()}개")
        
        # 종목별 데이터 수 출력
        for code in TARGET_STOCKS:
            cnt = len(df[df['stock_code'] == code])
            print(f"      {STOCK_NAMES[code]}: {cnt:,}건")
        
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
        print(f"\n🎯 타겟 재정의 (오늘 종가 → 내일 종가, 임계값: ±{self.target_threshold*100:.1f}%)")
        
        if 'next_rtn' not in df.columns:
            raise ValueError("next_rtn 컬럼이 없습니다.")
        
        df = df.dropna(subset=['next_rtn'])
        before_filter = len(df)
        
        # 임계값 기준 분류
        df_up = df[df['next_rtn'] >= self.target_threshold].copy()
        df_up['target'] = 1
        
        df_down = df[df['next_rtn'] <= -self.target_threshold].copy()
        df_down['target'] = 0
        
        df_filtered = pd.concat([df_up, df_down], ignore_index=True)
        df_filtered = df_filtered.sort_values(['날짜', 'stock_code']).reset_index(drop=True)
        
        # next_rtn 제거 (데이터 누수 방지)
        df_filtered = df_filtered.drop(columns=['next_rtn'])
        
        excluded = before_filter - len(df_filtered)
        print(f"   상승 (≥+{self.target_threshold*100:.1f}%): {len(df_up):,}건")
        print(f"   하락 (≤-{self.target_threshold*100:.1f}%): {len(df_down):,}건")
        print(f"   제외 (노이즈): {excluded:,}건 ({excluded/before_filter*100:.1f}%)")
        print(f"   필터링 후: {len(df_filtered):,}건")
        
        return df_filtered
    
    def add_lag_features(self, df):
        """Lag 피처 추가"""
        print(f"\n📊 Lag 피처 생성 중... (lag_days: {self.lag_days})")
        
        base_features = [
            'open_gap', 'high_ratio', 'low_ratio', 'volatility',
            '개인_체결강도', '외국인_체결강도', '기관계_체결강도',
            'vol_ratio', 'rsi'
        ]
        
        # 존재하는 피처만 사용
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
        
        # NaN 제거
        before_drop = len(df_with_lag)
        df_with_lag = df_with_lag.dropna()
        after_drop = len(df_with_lag)
        
        # ⚠️ TimeSeriesSplit을 위해 날짜 우선 정렬
        df_with_lag = df_with_lag.sort_values(['날짜', 'stock_code']).reset_index(drop=True)
        
        new_features = [c for c in df_with_lag.columns if '_lag' in c]
        print(f"   생성된 Lag 피처 수: {len(new_features)}개")
        print(f"   NaN 제거: {before_drop - after_drop:,}건 제거됨")
        
        return df_with_lag
    
    def get_features(self, df):
        """피처 컬럼 추출"""
        exclude_cols = ['날짜', 'target', 'stock_code', 'stock_code_encoded',
                        '시가', '고가', '저가', '종가', '거래량', '거래대금',
                        'stock_name', 'next_rtn']  # 문자열/타겟 관련 제외
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        return feature_cols
    
    def train_fold(self, X_train, y_train, X_val, y_val):
        """단일 Fold 학습 (Optuna 최적 파라미터 적용)"""
        models = {}
        
        # XGBoost (Optuna 최적화)
        xgb_model = xgb.XGBClassifier(
            n_estimators=741,
            max_depth=4,
            learning_rate=0.0156,
            subsample=0.7766,
            colsample_bytree=0.6643,
            reg_alpha=0.000133,
            reg_lambda=0.00135,
            min_child_weight=3,
            random_state=self.seed,
            eval_metric='auc',
            early_stopping_rounds=50
        )
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        models['xgboost'] = xgb_model
        
        # LightGBM (기본 파라미터, Optuna 후 업데이트)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=20,
            random_state=self.seed,
            verbose=-1
        )
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        models['lightgbm'] = lgb_model
        
        # CatBoost (Optuna 최적화)
        cat_model = CatBoostClassifier(
            iterations=665,
            depth=10,
            learning_rate=0.0124,
            l2_leaf_reg=1.637,
            bagging_temperature=0.582,
            random_strength=1.44e-07,
            random_state=self.seed,
            eval_metric='AUC',
            early_stopping_rounds=50,
            verbose=False
        )
        cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val)
        )
        models['catboost'] = cat_model
        
        return models
    
    def evaluate(self, y_true, y_pred, y_prob):
        """평가"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
        }
    
    def run(self):
        """학습 실행"""
        print("=" * 60)
        print("🚀 9개 종목 앙상블 학습 시작")
        print("=" * 60)
        
        # 데이터 로드
        df = self.load_data()
        
        # 피처 준비
        feature_cols = self.get_features(df)
        print(f"\n📊 피처 수: {len(feature_cols)}개")
        
        # stock_code 인코딩
        df['stock_code_encoded'] = self.label_encoder.fit_transform(df['stock_code'])
        if 'stock_code_encoded' not in feature_cols:
            feature_cols.append('stock_code_encoded')
        
        X = df[feature_cols].values
        y = df['target'].values
        
        # TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        all_metrics = {'xgboost': [], 'lightgbm': [], 'catboost': [], 'ensemble': []}
        
        print(f"\n📅 {self.n_splits}-Fold TimeSeriesSplit 학습")
        print("=" * 60)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\n[Fold {fold + 1}/{self.n_splits}]")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"   학습: {len(train_idx):,}건, 검증: {len(val_idx):,}건")
            
            # 학습
            fold_models = self.train_fold(X_train, y_train, X_val, y_val)
            
            # 평가
            for name, model in fold_models.items():
                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1]
                metrics = self.evaluate(y_val, y_pred, y_prob)
                all_metrics[name].append(metrics)
                print(f"   {name}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
            
            # 가중 앙상블 (3개 모델)
            xgb_prob = fold_models['xgboost'].predict_proba(X_val)[:, 1]
            lgb_prob = fold_models['lightgbm'].predict_proba(X_val)[:, 1]
            cat_prob = fold_models['catboost'].predict_proba(X_val)[:, 1]
            
            # 다양한 가중치 테스트 (xgb, lgb, cat)
            weights = [
                (0.33, 0.33, 0.34, 'equal'),      # 균등
                (0.2, 0.3, 0.5, 'cat_heavy'),     # CatBoost 중심
                (0.2, 0.2, 0.6, 'cat_only'),      # CatBoost 강조
                (0.3, 0.4, 0.3, 'lgb_heavy'),     # LightGBM 중심
                (0.0, 0.0, 1.0, 'cat_100'),       # CatBoost 단독
                (0.0, 1.0, 0.0, 'lgb_100'),       # LightGBM 단독
            ]
            
            best_auc = 0
            best_weight = None
            
            for w_xgb, w_lgb, w_cat, name in weights:
                y_prob_w = w_xgb * xgb_prob + w_lgb * lgb_prob + w_cat * cat_prob
                auc = roc_auc_score(y_val, y_prob_w)
                if auc > best_auc:
                    best_auc = auc
                    best_weight = (w_xgb, w_lgb, w_cat, name)
                    best_prob = y_prob_w
            
            # 기본 앙상블 (균등)
            y_prob_ensemble = (xgb_prob + lgb_prob + cat_prob) / 3
            y_pred_ensemble = (y_prob_ensemble >= 0.5).astype(int)
            metrics_ensemble = self.evaluate(y_val, y_pred_ensemble, y_prob_ensemble)
            all_metrics['ensemble'].append(metrics_ensemble)
            
            # 최적 가중치 앙상블
            y_pred_best = (best_prob >= 0.5).astype(int)
            metrics_best = self.evaluate(y_val, y_pred_best, best_prob)
            
            if 'best_ensemble' not in all_metrics:
                all_metrics['best_ensemble'] = []
            all_metrics['best_ensemble'].append(metrics_best)
            
            print(f"   ensemble(equal): AUC={metrics_ensemble['auc']:.4f}")
            print(f"   best({best_weight[3]}): AUC={metrics_best['auc']:.4f} ⭐")
            
            # 마지막 Fold 모델 저장
            if fold == self.n_splits - 1:
                self.models = fold_models
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("📋 교차 검증 결과 (평균 ± 표준편차)")
        print("=" * 60)
        
        for name in ['xgboost', 'lightgbm', 'catboost', 'ensemble', 'best_ensemble']:
            metrics_list = all_metrics[name]
            print(f"\n📈 {name.upper()}:")
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                values = [m[metric] for m in metrics_list]
                print(f"   {metric:10s}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        # 모델 저장
        self.save_models()
        
        return all_metrics
    
    def save_models(self):
        """모델 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "=" * 60)
        print("💾 모델 저장")
        print("=" * 60)
        
        for name, model in self.models.items():
            path = f"{self.model_dir}/{name}_{timestamp}.pkl"
            joblib.dump(model, path)
            print(f"   {name}: {path}")
        
        # LabelEncoder 저장
        le_path = f"{self.model_dir}/label_encoder_{timestamp}.pkl"
        joblib.dump(self.label_encoder, le_path)
        print(f"   label_encoder: {le_path}")
        
        # 종목 목록 저장
        stocks_path = f"{self.model_dir}/target_stocks_{timestamp}.txt"
        with open(stocks_path, 'w', encoding='utf-8') as f:
            for code in TARGET_STOCKS:
                f.write(f"{code},{STOCK_NAMES[code]}\n")
        print(f"   target_stocks: {stocks_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="9개 종목 앙상블 학습")
    parser.add_argument("--data", type=str, default="_data/merged_with_macro.csv",
                        help="데이터 경로 (기본: merged_with_macro.csv)")
    parser.add_argument("--no-macro", action="store_true", 
                        help="매크로 없이 기존 데이터 사용")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드 (기본: 42)")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="타겟 임계값 (기본: 0.01 = 1%%)")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="교차 검증 Fold 수 (기본: 5)")
    args = parser.parse_args()
    
    # 매크로 없이 실행할 경우
    if args.no_macro:
        data_path = "_data/merged_all_stocks_20260131.csv"
    else:
        data_path = args.data
    
    print(f"📂 데이터: {data_path}")
    
    trainer = Stock9Trainer(
        data_path=data_path,
        n_splits=args.n_splits,
        lag_days=[1, 3, 5],
        target_threshold=args.threshold,
        seed=args.seed
    )
    
    metrics = trainer.run()
    
    print("\n" + "=" * 60)
    print("✅ 학습 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

