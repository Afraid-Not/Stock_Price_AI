"""
실시간 예측 파이프라인
1. 최신 데이터 수집 (KIS API)
2. 전처리
3. 예측

사용법: python predict_realtime.py --date 2026-02-03
"""
import pandas as pd
import numpy as np
import joblib
import argparse
import os
import glob
from datetime import datetime, timedelta

from s00_get_token import get_access_token
from s01_kis_data_get import get_stock_daily_chart, get_investor_daily
from s02_rename import rename_file
from s03_preprocessing import StockPreprocessor


class RealtimePredictor:
    def __init__(self, model_dir='models', lag_days=[1, 2, 3, 5, 10]):
        self.model_dir = model_dir
        self.lag_days = lag_days
        self.models = {}
        self.label_encoder = None
        self.token = None
        
        # 종목 목록
        self.stocks = pd.read_csv("D:/stock/target_stocks.csv")
        self.stocks = self.stocks.dropna(subset=['Code'])
        self.stocks['Code'] = self.stocks['Code'].astype(int).astype(str).str.zfill(6)
        
    def load_models(self):
        """최신 모델 로드"""
        print("📂 모델 로드 중...")
        
        for model_name in ['xgboost', 'catboost']:
            pattern = f'{self.model_dir}/{model_name}_*.pkl'
            files = sorted(glob.glob(pattern), reverse=True)
            if files:
                self.models[model_name] = joblib.load(files[0])
                print(f"   ✅ {model_name}: {os.path.basename(files[0])}")
        
        le_files = sorted(glob.glob(f'{self.model_dir}/label_encoder_*.pkl'), reverse=True)
        if le_files:
            self.label_encoder = joblib.load(le_files[0])
            print(f"   ✅ LabelEncoder: {os.path.basename(le_files[0])}")
        
        if not self.models:
            raise FileNotFoundError("모델이 없습니다. train_ensemble.py를 먼저 실행하세요.")
    
    def collect_recent_data(self, end_date, days_back=100):
        """최근 N일 데이터 수집 (Lag 피처 생성에 필요)"""
        print(f"\n📥 최근 {days_back}일 데이터 수집 중...")
        
        self.token = get_access_token()
        if not self.token:
            raise Exception("토큰 발급 실패")
        
        end_dt = pd.to_datetime(end_date)
        start_dt = end_dt - timedelta(days=days_back + 10)  # 여유분
        
        start_str = start_dt.strftime("%Y%m%d")
        end_str = end_dt.strftime("%Y%m%d")
        
        all_data = []
        
        for _, row in self.stocks.iterrows():
            code = row['Code']
            name = row['Name']
            
            print(f"   {name}({code}) 수집 중...", end=" ")
            
            try:
                # 일봉 데이터
                df_chart = get_stock_daily_chart(self.token, code, start_str, end_str)
                
                if df_chart.empty:
                    print("❌ 데이터 없음")
                    continue
                
                # 투자자 데이터
                df_investor = get_investor_daily(self.token, code, end_str)
                
                # 병합
                if not df_investor.empty:
                    df_merged = pd.merge(df_chart, df_investor, on='stck_bsop_date', 
                                        how='left', suffixes=('', '_investor'))
                else:
                    df_merged = df_chart
                
                df_merged['stock_code'] = code
                df_merged['stock_name'] = name
                
                all_data.append(df_merged)
                print(f"✅ {len(df_merged)}건")
                
            except Exception as e:
                print(f"❌ 오류: {e}")
            
            import time
            time.sleep(0.5)  # API 제한
        
        if not all_data:
            raise Exception("수집된 데이터가 없습니다.")
        
        df_all = pd.concat(all_data, ignore_index=True)
        print(f"\n   총 수집: {len(df_all):,}건")
        
        return df_all
    
    def preprocess_data(self, df_raw):
        """데이터 전처리 (rename + preprocessing)"""
        print("\n⚙️ 데이터 전처리 중...")
        print(f"   원본 데이터: {len(df_raw)}건")
        
        # 임시 파일로 저장 후 처리
        temp_raw = '_data/temp_raw.csv'
        temp_renamed = '_data/temp_renamed.csv'
        temp_preprocessed = '_data/temp_preprocessed.csv'
        
        preprocessed_dfs = []
        
        # 종목별로 처리
        stock_codes = df_raw['stock_code'].unique() if 'stock_code' in df_raw.columns else []
        
        if len(stock_codes) == 0:
            print("   ❌ stock_code 컬럼이 없습니다.")
            return pd.DataFrame()
        
        for stock_code in stock_codes:
            stock_df = df_raw[df_raw['stock_code'] == stock_code].copy()
            stock_name = stock_df['stock_name'].iloc[0] if 'stock_name' in stock_df.columns else stock_code
            
            if stock_df.empty:
                continue
            
            print(f"   {stock_name}({stock_code}) 전처리 중...", end=" ")
            
            try:
                # stock_code, stock_name 제외하고 저장
                cols_to_save = [c for c in stock_df.columns if c not in ['stock_code', 'stock_name']]
                stock_df[cols_to_save].to_csv(temp_raw, index=False, encoding='utf-8-sig')
                
                # Rename
                rename_file(temp_raw, temp_renamed)
                
                # 전처리
                preprocessor = StockPreprocessor(stock_code=stock_code)
                df_processed = preprocessor.run_pipeline(temp_renamed, temp_preprocessed)
                
                if df_processed is not None and not df_processed.empty:
                    df_processed['stock_code'] = stock_code
                    df_processed['stock_name'] = stock_name
                    preprocessed_dfs.append(df_processed)
                    print(f"✅ {len(df_processed)}건")
                else:
                    print("❌ 빈 결과")
                    
            except Exception as e:
                print(f"❌ 오류: {e}")
        
        # 임시 파일 삭제
        for f in [temp_raw, temp_renamed, temp_preprocessed]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        
        if not preprocessed_dfs:
            print("   ❌ 전처리된 데이터가 없습니다.")
            return pd.DataFrame()
        
        df_preprocessed = pd.concat(preprocessed_dfs, ignore_index=True)
        print(f"\n   전처리 완료: {len(df_preprocessed):,}건")
        
        return df_preprocessed
    
    def add_lag_features(self, df):
        """Lag 피처 추가"""
        print("\n📊 Lag 피처 생성 중...")
        print(f"   입력 데이터: {len(df)}건, 컬럼: {list(df.columns)[:10]}...")
        
        if df.empty:
            print("   ❌ 입력 데이터가 비어있습니다.")
            return df
        
        if 'stock_code' not in df.columns:
            print("   ❌ stock_code 컬럼이 없습니다.")
            return df
        
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
            
            if stock_df.empty:
                continue
                
            stock_df = stock_df.sort_values('날짜')
            
            for lag in self.lag_days:
                for col in lag_cols:
                    if col in stock_df.columns:
                        stock_df[f'{col}_lag{lag}'] = stock_df[col].shift(lag)
            
            for col in ['gap_ma5', 'rsi', '외국인_체결강도', '기관계_체결강도']:
                if col in stock_df.columns:
                    stock_df[f'{col}_change'] = stock_df[col] - stock_df[col].shift(1)
            
            lag_dfs.append(stock_df)
        
        if not lag_dfs:
            print("   ❌ Lag 피처 생성 실패 - 데이터 없음")
            return df
        
        result = pd.concat(lag_dfs, ignore_index=True)
        print(f"   Lag 피처 생성 완료: {len(result)}건")
        return result
    
    def prepare_features(self, df):
        """피처 준비"""
        exclude_cols = ['날짜', 'target', 'stock_code', 'stock_name']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        if self.label_encoder is not None:
            stock_codes = df['stock_code'].values
            stock_code_encoded = []
            for code in stock_codes:
                try:
                    encoded = self.label_encoder.transform([code])[0]
                except ValueError:
                    encoded = -1
                stock_code_encoded.append(encoded)
            X['stock_code_encoded'] = np.array(stock_code_encoded)
        
        return X
    
    def predict(self, target_date):
        """전체 파이프라인 실행 및 예측"""
        print("=" * 60)
        print(f"🚀 실시간 예측 파이프라인")
        print(f"   예측 기준일: {target_date}")
        print("=" * 60)
        
        # 1. 모델 로드
        self.load_models()
        
        # 2. 최신 데이터 수집 (MA60 + Lag10 = 최소 70일 필요, 여유있게 100일)
        df_raw = self.collect_recent_data(target_date, days_back=100)
        
        # 3. 전처리
        df_preprocessed = self.preprocess_data(df_raw)
        
        # 4. Lag 피처 추가
        df_with_lag = self.add_lag_features(df_preprocessed)
        
        # 5. 타겟 날짜 데이터 추출
        df_with_lag['날짜'] = pd.to_datetime(df_with_lag['날짜'])
        target_dt = pd.to_datetime(target_date)
        
        df_target = df_with_lag[df_with_lag['날짜'] == target_dt].copy()
        
        if df_target.empty:
            # 가장 최근 날짜 사용
            latest_date = df_with_lag['날짜'].max()
            print(f"\n⚠️ {target_date} 데이터 없음, {latest_date.date()} 사용")
            df_target = df_with_lag[df_with_lag['날짜'] == latest_date].copy()
        
        # NaN 제거
        df_target = df_target.dropna()
        
        if df_target.empty:
            print("❌ 예측 가능한 데이터가 없습니다.")
            return None
        
        print(f"\n🎯 예측 대상: {len(df_target)}개 종목")
        
        # 6. 피처 준비 및 예측
        X = self.prepare_features(df_target)
        
        results = []
        for idx, row in df_target.iterrows():
            stock_code = row['stock_code']
            stock_name = row['stock_name']
            
            X_single = X.loc[[idx]]
            
            probas = []
            for name, model in self.models.items():
                try:
                    proba = model.predict_proba(X_single)[0][1]
                    probas.append(proba)
                except Exception as e:
                    print(f"   ⚠️ {name} 예측 오류: {e}")
            
            if probas:
                avg_proba = np.mean(probas)
                prediction = 1 if avg_proba >= 0.5 else 0
                
                results.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'prediction': prediction,
                    'probability': avg_proba,
                    'signal': '🔺 상승' if prediction == 1 else '🔻 하락'
                })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('probability', ascending=False)
        
        return results_df
    
    def display_results(self, results_df, top_n=10):
        """결과 출력 - 상승 확률 70%+, 60%+ 구분"""
        if results_df is None or results_df.empty:
            print("❌ 예측 결과가 없습니다.")
            return
        
        print("\n" + "=" * 70)
        print("📈 내일 예측 결과 (1% 이상 상승 예측)")
        print("=" * 70)
        
        # 상승 예측만 필터링
        up_stocks = results_df[results_df['prediction'] == 1].sort_values('probability', ascending=False)
        
        # 70% 이상
        up_70 = up_stocks[up_stocks['probability'] >= 0.70]
        # 60% 이상 70% 미만
        up_60 = up_stocks[(up_stocks['probability'] >= 0.60) & (up_stocks['probability'] < 0.70)]
        
        print(f"\n🔥 상승 확률 70% 이상 ({len(up_70)}개) - 강력 추천")
        print("-" * 60)
        if not up_70.empty:
            for _, row in up_70.iterrows():
                print(f"   ⭐ {str(row['stock_name']):12s} ({str(row['stock_code']).zfill(6)}) - {row['probability']*100:.1f}%")
        else:
            print("   없음")
        
        print(f"\n✅ 상승 확률 60~70% ({len(up_60)}개) - 추천")
        print("-" * 60)
        if not up_60.empty:
            for _, row in up_60.iterrows():
                print(f"   📌 {str(row['stock_name']):12s} ({str(row['stock_code']).zfill(6)}) - {row['probability']*100:.1f}%")
        else:
            print("   없음")
        
        # 요약
        print("\n" + "=" * 70)
        print("📋 요약")
        print("=" * 70)
        print(f"   총 종목: {len(results_df)}개")
        print(f"   🔥 70%+ 강력추천: {len(up_70)}개")
        print(f"   ✅ 60~70% 추천: {len(up_60)}개")
        print(f"   ⚪ 60% 미만: {len(up_stocks) - len(up_70) - len(up_60)}개 (생략)")
        
        if not up_70.empty:
            top = up_70.iloc[0]
            print(f"\n   🏆 TOP 추천: {top['stock_name']} ({str(top['stock_code']).zfill(6)}) - {top['probability']*100:.1f}%")
        elif not up_60.empty:
            top = up_60.iloc[0]
            print(f"\n   🏆 TOP 추천: {top['stock_name']} ({str(top['stock_code']).zfill(6)}) - {top['probability']*100:.1f}%")
        else:
            print("\n   ⚠️ 60% 이상 추천 종목 없음")
        
        print("=" * 70)
        
        return results_df


def main():
    parser = argparse.ArgumentParser(description='실시간 주식 예측')
    parser.add_argument('--date', '-d', type=str, default=None,
                        help='예측 기준일 (YYYY-MM-DD), 기본값: 오늘')
    parser.add_argument('--top', '-t', type=int, default=10,
                        help='상위 N개 출력')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='결과 저장 경로 (CSV)')
    
    args = parser.parse_args()
    
    # 기본값: 오늘
    if args.date is None:
        args.date = datetime.now().strftime('%Y-%m-%d')
    
    # 예측 실행
    predictor = RealtimePredictor()
    results = predictor.predict(args.date)
    
    # 결과 출력
    predictor.display_results(results, top_n=args.top)
    
    # 저장
    if args.output and results is not None:
        results.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"\n💾 저장: {args.output}")


if __name__ == "__main__":
    main()

