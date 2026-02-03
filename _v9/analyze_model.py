"""
모델 분석 및 최적화
- 피처 중요도 분석
- 최적 임계값 탐색
"""
import pandas as pd
import numpy as np
import joblib
import glob
import os

def load_latest_models(model_dir='models_9stocks'):
    """최신 모델 로드"""
    models = {}
    
    # XGBoost
    xgb_files = sorted(glob.glob(f"{model_dir}/xgboost_*.pkl"))
    if xgb_files:
        models['xgboost'] = joblib.load(xgb_files[-1])
        print(f"✅ XGBoost: {os.path.basename(xgb_files[-1])}")
    
    # CatBoost
    cat_files = sorted(glob.glob(f"{model_dir}/catboost_*.pkl"))
    if cat_files:
        models['catboost'] = joblib.load(cat_files[-1])
        print(f"✅ CatBoost: {os.path.basename(cat_files[-1])}")
    
    # LabelEncoder
    le_files = sorted(glob.glob(f"{model_dir}/label_encoder_*.pkl"))
    if le_files:
        models['label_encoder'] = joblib.load(le_files[-1])
        print(f"✅ LabelEncoder: {os.path.basename(le_files[-1])}")
    
    return models


def analyze_feature_importance(models, top_n=20):
    """피처 중요도 분석"""
    print("\n" + "=" * 60)
    print("📊 피처 중요도 분석")
    print("=" * 60)
    
    importance_df = pd.DataFrame()
    
    # XGBoost
    if 'xgboost' in models:
        xgb_model = models['xgboost']
        xgb_imp = pd.DataFrame({
            'feature': xgb_model.feature_names_in_ if hasattr(xgb_model, 'feature_names_in_') else [f'f{i}' for i in range(len(xgb_model.feature_importances_))],
            'xgb_importance': xgb_model.feature_importances_
        })
        importance_df = xgb_imp
    
    # CatBoost
    if 'catboost' in models:
        cat_model = models['catboost']
        cat_imp = cat_model.feature_importances_
        if importance_df.empty:
            importance_df = pd.DataFrame({
                'feature': [f'f{i}' for i in range(len(cat_imp))],
                'cat_importance': cat_imp
            })
        else:
            importance_df['cat_importance'] = cat_imp
    
    # 평균 중요도
    if 'xgb_importance' in importance_df.columns and 'cat_importance' in importance_df.columns:
        importance_df['avg_importance'] = (importance_df['xgb_importance'] + importance_df['cat_importance']) / 2
    elif 'xgb_importance' in importance_df.columns:
        importance_df['avg_importance'] = importance_df['xgb_importance']
    else:
        importance_df['avg_importance'] = importance_df['cat_importance']
    
    # 정렬
    importance_df = importance_df.sort_values('avg_importance', ascending=False).reset_index(drop=True)
    
    # 상위 피처
    print(f"\n🔝 상위 {top_n}개 피처:")
    print("-" * 50)
    for i, row in importance_df.head(top_n).iterrows():
        bar = "█" * int(row['avg_importance'] / importance_df['avg_importance'].max() * 20)
        print(f"{i+1:2d}. {row['feature']:30s} {row['avg_importance']:.4f} {bar}")
    
    # 하위 피처 (제거 후보)
    print(f"\n🔻 하위 10개 피처 (제거 후보):")
    print("-" * 50)
    for i, row in importance_df.tail(10).iterrows():
        print(f"    {row['feature']:30s} {row['avg_importance']:.4f}")
    
    # 저장
    importance_df.to_csv('models_9stocks/feature_importance.csv', index=False)
    print(f"\n💾 저장: models_9stocks/feature_importance.csv")
    
    return importance_df


def suggest_optimizations(importance_df):
    """최적화 제안"""
    print("\n" + "=" * 60)
    print("💡 최적화 제안")
    print("=" * 60)
    
    # 1. 제거 추천 피처 (중요도 하위 20%)
    threshold = importance_df['avg_importance'].quantile(0.2)
    low_importance = importance_df[importance_df['avg_importance'] < threshold]['feature'].tolist()
    
    print(f"\n1️⃣ 제거 추천 피처 (하위 20%, {len(low_importance)}개):")
    for f in low_importance:
        print(f"   - {f}")
    
    # 2. 핵심 피처 (상위 10개)
    top_features = importance_df.head(10)['feature'].tolist()
    print(f"\n2️⃣ 핵심 피처 (상위 10개):")
    for f in top_features:
        print(f"   ✅ {f}")
    
    # 3. Lag 피처 분석
    lag_features = [f for f in importance_df['feature'] if '_lag' in f]
    lag_importance = importance_df[importance_df['feature'].isin(lag_features)]['avg_importance'].sum()
    total_importance = importance_df['avg_importance'].sum()
    
    print(f"\n3️⃣ Lag 피처 기여도:")
    print(f"   Lag 피처 수: {len(lag_features)}개")
    print(f"   중요도 비중: {lag_importance/total_importance*100:.1f}%")
    
    # 4. 매크로 피처 분석
    macro_features = ['kospi_return', 'kospi_gap_ma5', 'kospi_volatility', 
                      'usdkrw_return', 'usdkrw_gap_ma5']
    macro_in_data = [f for f in macro_features if f in importance_df['feature'].values]
    
    if macro_in_data:
        print(f"\n4️⃣ 매크로 피처 기여도:")
        for f in macro_in_data:
            imp = importance_df[importance_df['feature'] == f]['avg_importance'].values[0]
            rank = importance_df[importance_df['feature'] == f].index[0] + 1
            print(f"   {f}: 중요도 {imp:.4f} (순위 {rank})")


if __name__ == "__main__":
    print("=" * 60)
    print("🔍 모델 분석 및 최적화")
    print("=" * 60)
    
    # 모델 로드
    models = load_latest_models()
    
    if models:
        # 피처 중요도 분석
        importance_df = analyze_feature_importance(models)
        
        # 최적화 제안
        suggest_optimizations(importance_df)
    else:
        print("❌ 모델을 찾을 수 없습니다.")

