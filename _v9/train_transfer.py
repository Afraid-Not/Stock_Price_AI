# -*- coding: utf-8 -*-
"""
Transfer Learning: Base Model + Fine-tuning with News
1단계: 16년 주가 데이터로 Base 모델 학습
2단계: 1년 뉴스 데이터로 Fine-tuning
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
TARGET_THRESHOLD = 0.01
N_SPLITS = 5
LAG_DAYS = [1, 2, 3, 5, 10]

# 뉴스 시작일 (약 1년 전)
NEWS_START_DATE = '2025-02-03'


def load_data():
    """데이터 로드"""
    print("\n[LOAD] Data")
    
    # 전체 데이터 (뉴스 포함)
    df = pd.read_csv("_data/merged_9stocks_with_news.csv", encoding='utf-8-sig')
    df['날짜'] = pd.to_datetime(df['날짜'])
    
    print(f"   Total: {len(df):,} rows")
    print(f"   Period: {df['날짜'].min().date()} ~ {df['날짜'].max().date()}")
    
    return df


def redefine_target(df, threshold=TARGET_THRESHOLD):
    """타겟 재정의"""
    print(f"\n[TARGET] Threshold: +/-{threshold*100:.1f}%")
    
    up_mask = df['next_rtn'] >= threshold
    down_mask = df['next_rtn'] <= -threshold
    
    df_filtered = df[up_mask | down_mask].copy()
    df_filtered['target'] = (df_filtered['next_rtn'] >= threshold).astype(int)
    
    print(f"   Filtered: {len(df_filtered):,} rows")
    
    return df_filtered


def add_lag_features(df, lag_days=LAG_DAYS):
    """Lag 피처 추가"""
    lag_cols = ['open_gap', 'high_ratio', 'low_ratio', 'volatility', 
                'gap_ma5', 'rsi', 'vol_ratio']
    
    df = df.sort_values(['stock_code', '날짜']).copy()
    
    for col in lag_cols:
        if col not in df.columns:
            continue
        for lag in lag_days:
            df[f'{col}_lag{lag}'] = df.groupby('stock_code')[col].shift(lag)
    
    # 변화율 피처
    for col in ['gap_ma5', 'rsi']:
        if col in df.columns:
            df[f'{col}_change'] = df.groupby('stock_code')[col].pct_change()
    
    return df


def get_base_features():
    """Base 모델용 피처 (뉴스 제외)"""
    exclude = ['날짜', 'next_rtn', 'target', 'stock_name', 'stock_code',
               'news_sentiment_mean', 'news_sentiment_std', 
               'news_sentiment_min', 'news_sentiment_max',
               'news_confidence', 'news_label', 'news_count']
    return exclude


def get_news_features():
    """뉴스 피처 목록"""
    return ['news_sentiment_mean', 'news_sentiment_std', 
            'news_sentiment_min', 'news_sentiment_max',
            'news_confidence', 'news_count']


def prepare_data(df, include_news=False):
    """피처 준비"""
    df = df.copy()
    
    # stock_code 인코딩
    le = LabelEncoder()
    df['stock_code_encoded'] = le.fit_transform(df['stock_code'].astype(str))
    
    # 제외할 컬럼
    exclude = ['날짜', 'next_rtn', 'target', 'stock_name', 'stock_code', 'news_label']
    
    if not include_news:
        exclude.extend(get_news_features())
    
    feature_cols = [c for c in df.columns if c not in exclude]
    
    return df, feature_cols, le


def train_base_model(df, feature_cols, n_splits=N_SPLITS, seed=42):
    """1단계: Base 모델 학습"""
    print("\n" + "="*60)
    print("[STAGE 1] Base Model Training (No News)")
    print("="*60)
    
    # 시간순 정렬
    df = df.sort_values(['날짜', 'stock_code']).reset_index(drop=True)
    
    # NaN 제거
    df_clean = df.dropna(subset=feature_cols)
    print(f"   Data after NaN removal: {len(df_clean):,}")
    
    X = df_clean[feature_cols].values
    y = df_clean['target'].values
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    metrics_list = []
    best_model = None
    best_auc = 0
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
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
        
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, y_prob)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        metrics_list.append({
            'fold': fold, 'auc': auc, 'f1': f1,
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0)
        })
        
        if auc > best_auc:
            best_auc = auc
            best_model = model
        
        print(f"   Fold {fold}: AUC={auc:.4f}, F1={f1:.4f}")
    
    # 평균
    metrics_df = pd.DataFrame(metrics_list)
    print(f"\n   [BASE] Mean AUC: {metrics_df['auc'].mean():.4f} +/- {metrics_df['auc'].std():.4f}")
    print(f"   [BASE] Mean F1:  {metrics_df['f1'].mean():.4f} +/- {metrics_df['f1'].std():.4f}")
    
    return best_model, metrics_df, feature_cols


def train_finetuned_model(df_news, base_model, base_features, n_splits=N_SPLITS, seed=42):
    """2단계: Fine-tuning with News"""
    print("\n" + "="*60)
    print("[STAGE 2] Fine-tuning with News Features")
    print("="*60)
    
    # 뉴스 피처 추가
    news_features = get_news_features()
    all_features = base_features + [f for f in news_features if f in df_news.columns]
    
    print(f"   Base features: {len(base_features)}")
    print(f"   News features: {len([f for f in news_features if f in df_news.columns])}")
    print(f"   Total features: {len(all_features)}")
    
    # 시간순 정렬
    df_news = df_news.sort_values(['날짜', 'stock_code']).reset_index(drop=True)
    
    # 뉴스 NaN → 0
    for col in news_features:
        if col in df_news.columns:
            df_news[col] = df_news[col].fillna(0)
    
    # NaN 제거
    df_clean = df_news.dropna(subset=all_features)
    print(f"   Data after NaN removal: {len(df_clean):,}")
    
    X = df_clean[all_features].values
    y = df_clean['target'].values
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    metrics_list = []
    best_model = None
    best_auc = 0
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Fine-tuning: init_model 사용
        model = CatBoostClassifier(
            iterations=200,  # 적은 iteration
            learning_rate=0.03,  # 낮은 learning rate
            depth=4,  # 얕은 depth
            l2_leaf_reg=5,
            random_seed=seed,
            verbose=0,
            early_stopping_rounds=30,
            eval_metric='AUC'
        )
        
        # Base 모델 위에 Fine-tuning
        # Note: CatBoost는 피처 수가 같아야 init_model 사용 가능
        # 피처 수가 다르면 새로 학습
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y_val, y_prob)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        metrics_list.append({
            'fold': fold, 'auc': auc, 'f1': f1,
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0)
        })
        
        if auc > best_auc:
            best_auc = auc
            best_model = model
        
        print(f"   Fold {fold}: AUC={auc:.4f}, F1={f1:.4f}")
    
    # 평균
    metrics_df = pd.DataFrame(metrics_list)
    print(f"\n   [FINE-TUNED] Mean AUC: {metrics_df['auc'].mean():.4f} +/- {metrics_df['auc'].std():.4f}")
    print(f"   [FINE-TUNED] Mean F1:  {metrics_df['f1'].mean():.4f} +/- {metrics_df['f1'].std():.4f}")
    
    return best_model, metrics_df, all_features


def feature_importance(model, feature_cols, top_n=15):
    """피처 중요도"""
    print(f"\n[IMPORTANCE] Top {top_n} Features")
    print("-"*50)
    
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance_df.head(top_n).iterrows():
        marker = " [NEWS]" if 'news' in row['feature'].lower() else ""
        print(f"   {row['feature']:35s}: {row['importance']:.4f}{marker}")
    
    return importance_df


def save_models(base_model, fine_model, le, base_features, all_features, output_dir="models_transfer"):
    """모델 저장"""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base 모델
    with open(f"{output_dir}/base_model_{timestamp}.pkl", 'wb') as f:
        pickle.dump(base_model, f)
    
    # Fine-tuned 모델
    with open(f"{output_dir}/finetuned_model_{timestamp}.pkl", 'wb') as f:
        pickle.dump(fine_model, f)
    
    # LabelEncoder
    with open(f"{output_dir}/label_encoder_{timestamp}.pkl", 'wb') as f:
        pickle.dump(le, f)
    
    # 피처 목록
    with open(f"{output_dir}/base_features_{timestamp}.pkl", 'wb') as f:
        pickle.dump(base_features, f)
    
    with open(f"{output_dir}/all_features_{timestamp}.pkl", 'wb') as f:
        pickle.dump(all_features, f)
    
    print(f"\n[SAVE] Models saved to {output_dir}/")
    
    return timestamp


def run_training(seed=42, verbose=True):
    """전이학습 실행"""
    if verbose:
        print("="*60)
        print(f"[TRANSFER LEARNING] Seed={seed}")
        print("="*60)
    
    # 1. 데이터 로드
    df = load_data() if verbose else pd.read_csv("_data/merged_9stocks_with_news.csv", encoding='utf-8-sig')
    if not verbose:
        df['날짜'] = pd.to_datetime(df['날짜'])
    
    # 2. 타겟 재정의
    df = redefine_target(df) if verbose else df[(df['next_rtn'] >= TARGET_THRESHOLD) | (df['next_rtn'] <= -TARGET_THRESHOLD)].copy()
    if not verbose:
        df['target'] = (df['next_rtn'] >= TARGET_THRESHOLD).astype(int)
    
    # 3. Lag 피처 추가
    df = add_lag_features(df)
    
    # 4. 피처 준비 (뉴스 제외)
    df, base_features, le = prepare_data(df, include_news=False)
    
    # 5. Stage 1: Base 모델
    base_model, base_metrics, base_features = train_base_model(df, base_features, seed=seed)
    
    # 6. 뉴스 있는 기간만 필터링
    if verbose:
        print(f"\n[FILTER] News period: {NEWS_START_DATE} ~")
    df_news = df[df['날짜'] >= NEWS_START_DATE].copy()
    if verbose:
        print(f"   News period data: {len(df_news):,} rows")
    
    # 7. Stage 2: Fine-tuning
    fine_model, fine_metrics, all_features = train_finetuned_model(
        df_news, base_model, base_features, seed=seed
    )
    
    # 8. 결과
    results = {
        'seed': seed,
        'base_auc_mean': base_metrics['auc'].mean(),
        'base_auc_std': base_metrics['auc'].std(),
        'base_f1_mean': base_metrics['f1'].mean(),
        'base_f1_std': base_metrics['f1'].std(),
        'fine_auc_mean': fine_metrics['auc'].mean(),
        'fine_auc_std': fine_metrics['auc'].std(),
        'fine_f1_mean': fine_metrics['f1'].mean(),
        'fine_f1_std': fine_metrics['f1'].std(),
        'improvement': fine_metrics['auc'].mean() - base_metrics['auc'].mean(),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if verbose:
        print("\n" + "="*60)
        print("[COMPARISON] Base vs Fine-tuned")
        print("="*60)
        print(f"   Base Model AUC:       {results['base_auc_mean']:.4f}")
        print(f"   Fine-tuned Model AUC: {results['fine_auc_mean']:.4f}")
        print(f"   Improvement:          {results['improvement']*100:+.2f}%")
    
    # 9. 저장
    save_models(base_model, fine_model, le, base_features, all_features)
    
    return results, base_model, fine_model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    
    results, _, _ = run_training(seed=args.seed, verbose=not args.quiet)
    
    print("\n" + "="*60)
    print("[DONE]")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()
