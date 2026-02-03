"""
학습된 모델로 종목별 예측 수행
사용법: python predict.py --date 2026-01-30
"""
import pandas as pd
import numpy as np
import joblib
import argparse
import os
import glob
from datetime import datetime, timedelta


class StockPredictor:
    def __init__(self, model_dir='models', data_path='_data/merged_all_stocks_20260131.csv',
                 lag_days=[1, 2, 3, 5, 10]):
        self.model_dir = model_dir
        self.data_path = data_path
        self.lag_days = lag_days
        self.models = {}
        self.label_encoder = None
        
    def load_models(self):
        """최신 모델 로드"""
        print("📂 모델 로드 중...")
        
        # 최신 모델 파일 찾기
        for model_name in ['xgboost', 'catboost']:
            pattern = f'{self.model_dir}/{model_name}_*.pkl'
            files = sorted(glob.glob(pattern), reverse=True)
            if files:
                self.models[model_name] = joblib.load(files[0])
                print(f"   ✅ {model_name}: {os.path.basename(files[0])}")
            else:
                print(f"   ❌ {model_name}: 모델 없음")
        
        # Label Encoder 로드
        le_files = sorted(glob.glob(f'{self.model_dir}/label_encoder_*.pkl'), reverse=True)
        if le_files:
            self.label_encoder = joblib.load(le_files[0])
            print(f"   ✅ LabelEncoder: {os.path.basename(le_files[0])}")
        
        if not self.models:
            raise FileNotFoundError("로드된 모델이 없습니다. 먼저 train_ensemble.py를 실행하세요.")
    
    def load_data(self):
        """데이터 로드"""
        print("\n📂 데이터 로드 중...")
        df = pd.read_csv(self.data_path)
        df['날짜'] = pd.to_datetime(df['날짜'])
        df = df.sort_values(['stock_code', '날짜']).reset_index(drop=True)
        print(f"   총 데이터: {len(df):,}건")
        return df
    
    def add_lag_features(self, df):
        """Lag 피처 추가 (학습 때와 동일)"""
        lag_cols = [
            'open_gap', 'high_ratio', 'low_ratio', 'volatility',
            'gap_ma5', 'gap_ma20', 'gap_ma60',
            '개인_체결강도', '외국인_체결강도', '기관계_체결강도',
            'vol_ratio', 'vol_ma5_ratio', 'rsi',
            'macd_ratio', 'macd_diff_ratio',
            'bb_upper_ratio', 'bb_lower_ratio'
        ]
        
        lag_dfs = []
        for stock_code in df['stock_code'].unique():
            stock_df = df[df['stock_code'] == stock_code].copy()
            stock_df = stock_df.sort_values('날짜')
            
            for lag in self.lag_days:
                for col in lag_cols:
                    if col in stock_df.columns:
                        stock_df[f'{col}_lag{lag}'] = stock_df[col].shift(lag)
            
            for col in ['gap_ma5', 'rsi', '외국인_체결강도', '기관계_체결강도']:
                if col in stock_df.columns:
                    stock_df[f'{col}_change'] = stock_df[col] - stock_df[col].shift(1)
            
            lag_dfs.append(stock_df)
        
        return pd.concat(lag_dfs, ignore_index=True)
    
    def prepare_features(self, df):
        """피처 준비 (학습 때와 동일)"""
        exclude_cols = ['날짜', 'target', 'stock_code', 'stock_name']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        # 종목 코드 인코딩
        if self.label_encoder is not None:
            stock_codes = df['stock_code'].values
            # 새로운 종목 코드가 있으면 -1로 처리
            try:
                stock_code_encoded = self.label_encoder.transform(stock_codes)
            except ValueError:
                # 알려지지 않은 종목은 -1
                stock_code_encoded = []
                for code in stock_codes:
                    try:
                        encoded = self.label_encoder.transform([code])[0]
                    except ValueError:
                        encoded = -1
                    stock_code_encoded.append(encoded)
                stock_code_encoded = np.array(stock_code_encoded)
            
            X['stock_code_encoded'] = stock_code_encoded
        
        return X
    
    def predict(self, target_date):
        """특정 날짜 기준 예측"""
        # 모델 로드
        self.load_models()
        
        # 데이터 로드
        df = self.load_data()
        
        # Lag 피처 추가
        print("\n📊 Lag 피처 생성 중...")
        df = self.add_lag_features(df)
        
        # 날짜 변환
        target_date = pd.to_datetime(target_date)
        print(f"\n🎯 예측 기준일: {target_date.date()}")
        
        # 해당 날짜의 데이터 필터링
        df_target = df[df['날짜'] == target_date].copy()
        
        if df_target.empty:
            # 해당 날짜가 없으면 가장 가까운 이전 날짜 사용
            available_dates = df['날짜'].unique()
            past_dates = [d for d in available_dates if d <= target_date]
            if past_dates:
                closest_date = max(past_dates)
                print(f"   ⚠️ {target_date.date()} 데이터 없음, {pd.Timestamp(closest_date).date()} 사용")
                df_target = df[df['날짜'] == closest_date].copy()
            else:
                print(f"   ❌ 사용 가능한 데이터가 없습니다.")
                return None
        
        # NaN 제거
        df_target = df_target.dropna()
        
        if df_target.empty:
            print("   ❌ NaN 제거 후 데이터가 없습니다.")
            return None
        
        print(f"   예측 대상 종목: {len(df_target)}개")
        
        # 피처 준비
        X = self.prepare_features(df_target)
        
        # 예측
        print("\n🔮 예측 수행 중...")
        results = []
        
        for idx, row in df_target.iterrows():
            stock_code = row['stock_code']
            stock_name = row['stock_name']
            
            X_single = X.loc[[idx]]
            
            # 각 모델 예측
            probas = []
            for name, model in self.models.items():
                proba = model.predict_proba(X_single)[0][1]
                probas.append(proba)
            
            # 앙상블 (평균)
            avg_proba = np.mean(probas)
            prediction = 1 if avg_proba >= 0.5 else 0
            
            results.append({
                'stock_code': stock_code,
                'stock_name': stock_name,
                'prediction': prediction,
                'probability': avg_proba,
                'signal': '🔺 상승' if prediction == 1 else '🔻 하락'
            })
        
        # 결과 정리
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('probability', ascending=False)
        
        return results_df
    
    def display_results(self, results_df, top_n=None):
        """결과 출력"""
        if results_df is None or results_df.empty:
            print("❌ 예측 결과가 없습니다.")
            return
        
        print("\n" + "=" * 70)
        print("📈 예측 결과 (내일 1% 이상 상승/하락 예측)")
        print("=" * 70)
        
        # 상승 예측 종목
        up_stocks = results_df[results_df['prediction'] == 1].sort_values('probability', ascending=False)
        down_stocks = results_df[results_df['prediction'] == 0].sort_values('probability', ascending=True)
        
        print(f"\n🔺 상승 예측 종목 ({len(up_stocks)}개)")
        print("-" * 50)
        if not up_stocks.empty:
            for _, row in up_stocks.head(top_n).iterrows():
                print(f"   {row['stock_name']:12s} ({row['stock_code']}) - 확률: {row['probability']*100:.1f}%")
        else:
            print("   없음")
        
        print(f"\n🔻 하락 예측 종목 ({len(down_stocks)}개)")
        print("-" * 50)
        if not down_stocks.empty:
            for _, row in down_stocks.head(top_n).iterrows():
                print(f"   {row['stock_name']:12s} ({row['stock_code']}) - 확률: {(1-row['probability'])*100:.1f}%")
        else:
            print("   없음")
        
        # 요약
        print("\n" + "=" * 70)
        print("📋 요약")
        print("=" * 70)
        print(f"   총 종목: {len(results_df)}개")
        print(f"   상승 예측: {len(up_stocks)}개")
        print(f"   하락 예측: {len(down_stocks)}개")
        
        # 상위 추천
        if not up_stocks.empty:
            top_pick = up_stocks.iloc[0]
            print(f"\n   ⭐ TOP 추천: {top_pick['stock_name']} ({top_pick['stock_code']}) - {top_pick['probability']*100:.1f}%")
        
        return results_df


def main():
    parser = argparse.ArgumentParser(description='주식 예측')
    parser.add_argument('--date', '-d', type=str, required=True,
                        help='예측 기준일 (YYYY-MM-DD 형식, 예: 2026-01-30)')
    parser.add_argument('--top', '-t', type=int, default=10,
                        help='상위 N개만 출력 (기본값: 10)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='결과 저장 경로 (CSV)')
    
    args = parser.parse_args()
    
    # 예측 실행
    predictor = StockPredictor()
    results = predictor.predict(args.date)
    
    # 결과 출력
    predictor.display_results(results, top_n=args.top)
    
    # 결과 저장
    if args.output and results is not None:
        results.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"\n💾 결과 저장: {args.output}")


if __name__ == "__main__":
    main()

