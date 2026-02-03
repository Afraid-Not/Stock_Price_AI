# -*- coding: utf-8 -*-
"""
9종목 + 뉴스 sentiment 모델 학습
CatBoost 기반
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.stdout.reconfigure(encoding='utf-8')

# 설정
RANDOM_SEED = 42
TARGET_THRESHOLD = 0.01
N_SPLITS = 5
LAG_DAYS = [1, 2, 3, 5, 10]

def load_data(data_path="_data/merged_9stocks_with_news.csv"):
    """데이터 로드 및 전처리"""
    print(f"\n[LOAD] {data_path}")
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    df['날짜'] = pd.to_datetime(df['날짜'])
    
    # 뉴스 피처 NaN → 0 채우기
    news_cols = ['news_sentiment_mean', 'news_sentiment_std', 
                 'news_sentiment_min', 'news_sentiment_max',
                 'news_confidence', 'news_count']
    for col in news_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    print(f"   Total: {len(df):,} rows")
    print(f"   Period: {df['날짜'].min().date()} ~ {df['날짜'].max().date()}")
    print(f"   Stocks: {df['stock_code'].nunique()}")
    
    return df

def redefine_target(df, threshold=TARGET_THRESHOLD):
    """타겟 재정의: 1% 이상 움직임만"""
    print(f"\n[TARGET] Threshold: +/-{threshold*100:.1f}%")
    
    up_mask = df['next_rtn'] >= threshold
    down_mask = df['next_rtn'] <= -threshold
    
    df_filtered = df[up_mask | down_mask].copy()
    df_filtered['target'] = (df_filtered['next_rtn'] >= threshold).astype(int)
    
    print(f"   Up (>=+{threshold*100:.1f}%): {up_mask.sum():,}")
    print(f"   Down (<=-{threshold*100:.1f}%): {down_mask.sum():,}")
    print(f"   Filtered: {len(df) - len(df_filtered):,} ({(len(df)-len(df_filtered))/len(df)*100:.1f}%)")
    print(f"   Remaining: {len(df_filtered):,}")
    
    return df_filtered

def add_lag_features(df, lag_days=LAG_DAYS):
    """Lag 피처 추가 (뉴스 제외)"""
    print(f"\n[LAG] Creating lag features: {lag_days}")
    
    # 뉴스 피처는 lag 제외!
    lag_cols = ['open_gap', 'high_ratio', 'low_ratio', 'volatility', 
                'gap_ma5', 'rsi', 'vol_ratio']
    
    df = df.sort_values(['stock_code', '날짜']).copy()
    
    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in lag_days:
            df[f'{col}_lag{lag}'] = df.groupby('stock_code')[col].shift(lag)
    
    # 변화율 피처 (뉴스 제외)
    change_cols = ['gap_ma5', 'rsi']
    for col in change_cols:
        if col not in df.columns:
            continue
        df[f'{col}_change'] = df.groupby('stock_code')[col].pct_change()
    
    # NaN 제거
    before = len(df)
    df = df.dropna()
    print(f"   NaN removed: {before - len(df):,}")
    print(f"   Final: {len(df):,}")
    
    return df

def prepare_features(df):
    """피처 준비"""
    
    exclude_cols = ['날짜', 'next_rtn', 'target', 'stock_name', 'news_label']
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # stock_code 인코딩
    le = LabelEncoder()
    df['stock_code_encoded'] = le.fit_transform(df['stock_code'].astype(str))
    
    feature_cols = [c for c in feature_cols if c != 'stock_code']
    feature_cols.append('stock_code_encoded')
    
    print(f"\n[FEATURES] Total: {len(feature_cols)}")
    
    # 뉴스 피처 확인
    news_features = [c for c in feature_cols if 'news' in c.lower()]
    print(f"   News features: {news_features}")
    
    return df, feature_cols, le

def train_model(df, feature_cols, n_splits=N_SPLITS, seed=RANDOM_SEED):
    """TimeSeriesSplit으로 학습"""
    print(f"\n[TRAIN] TimeSeriesSplit (n_splits={n_splits}, seed={seed})")
    
    # 시간순 정렬 (중요!)
    df = df.sort_values(['날짜', 'stock_code']).reset_index(drop=True)
    
    X = df[feature_cols].values
    y = df['target'].values
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    metrics_list = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # CatBoost
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            random_seed=seed,
            verbose=0,
            early_stopping_rounds=50,
            eval_metric='AUC'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        
        # 예측
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # 메트릭
        metrics = {
            'fold': fold,
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'auc': roc_auc_score(y_val, y_prob)
        }
        metrics_list.append(metrics)
        models.append(model)
        
        print(f"   Fold {fold}: AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}")
    
    return models, metrics_list

def print_results(metrics_list):
    """결과 출력"""
    print("\n" + "="*60)
    print("[RESULTS] Cross-Validation Summary")
    print("="*60)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    for col in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        mean = metrics_df[col].mean()
        std = metrics_df[col].std()
        print(f"   {col:12s}: {mean:.4f} +/- {std:.4f}")
    
    return metrics_df

def save_model(models, le, feature_cols, output_dir="models_news"):
    """모델 저장"""
    Path(output_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 가장 좋은 모델 저장 (마지막 fold)
    model_path = f"{output_dir}/catboost_news_{timestamp}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(models[-1], f)
    
    # LabelEncoder 저장
    le_path = f"{output_dir}/label_encoder_{timestamp}.pkl"
    with open(le_path, 'wb') as f:
        pickle.dump(le, f)
    
    # 피처 목록 저장
    features_path = f"{output_dir}/features_{timestamp}.pkl"
    with open(features_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print(f"\n[SAVE] {model_path}")
    
    return model_path

def feature_importance(models, feature_cols, top_n=15):
    """피처 중요도"""
    print(f"\n[IMPORTANCE] Top {top_n} Features")
    print("-"*40)
    
    # 평균 중요도
    importances = np.mean([m.feature_importances_ for m in models], axis=0)
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(top_n).iterrows():
        marker = " [NEWS]" if 'news' in row['feature'].lower() else ""
        print(f"   {row['feature']:30s}: {row['importance']:.4f}{marker}")
    
    return importance_df

def main():
    print("="*60)
    print("[TRAIN] 9 Stocks + News Sentiment Model")
    print("="*60)
    
    # 1. 데이터 로드
    df = load_data()
    
    # 2. 타겟 재정의
    df = redefine_target(df)
    
    # 3. Lag 피처 추가
    df = add_lag_features(df)
    
    # 4. 피처 준비
    df, feature_cols, le = prepare_features(df)
    
    # 5. 학습
    models, metrics_list = train_model(df, feature_cols)
    
    # 6. 결과
    metrics_df = print_results(metrics_list)
    
    # 7. 피처 중요도
    importance_df = feature_importance(models, feature_cols)
    
    # 8. 저장
    save_model(models, le, feature_cols)
    
    print("\n" + "="*60)
    print("[DONE]")
    print("="*60)
    
    return metrics_df, importance_df

if __name__ == "__main__":
    main()

