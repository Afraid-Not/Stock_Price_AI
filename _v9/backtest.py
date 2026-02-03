"""
예측 결과 백테스트
특정 날짜 예측 → 다음날 실제 결과와 비교

사용법: python backtest.py --date 2026-01-30
"""
import pandas as pd
import numpy as np
import joblib
import argparse
import glob
import os


class Backtester:
    def __init__(self, model_dir='models', data_path='_data/merged_all_stocks_20260131.csv',
                 lag_days=[1, 2, 3, 5, 10], target_threshold=0.01):
        self.model_dir = model_dir
        self.data_path = data_path
        self.lag_days = lag_days
        self.target_threshold = target_threshold
        self.models = {}
        self.label_encoder = None
        
    def load_models(self):
        """모델 로드"""
        print("📂 모델 로드 중...")
        for model_name in ['xgboost', 'catboost']:
            pattern = f'{self.model_dir}/{model_name}_*.pkl'
            files = sorted(glob.glob(pattern), reverse=True)
            if files:
                self.models[model_name] = joblib.load(files[0])
                print(f"   ✅ {model_name}")
        
        le_files = sorted(glob.glob(f'{self.model_dir}/label_encoder_*.pkl'), reverse=True)
        if le_files:
            self.label_encoder = joblib.load(le_files[0])
    
    def load_data(self):
        """데이터 로드 및 Lag 피처 추가"""
        print("\n📂 데이터 로드 중...")
        df = pd.read_csv(self.data_path)
        df['날짜'] = pd.to_datetime(df['날짜'])
        df = df.sort_values(['stock_code', '날짜']).reset_index(drop=True)
        
        # 다음날 수익률 계산 (실제 결과 확인용)
        returns = []
        for stock_code in df['stock_code'].unique():
            stock_df = df[df['stock_code'] == stock_code].copy()
            stock_df = stock_df.sort_values('날짜')
            stock_df['next_return'] = stock_df['open_gap'].shift(-1)
            returns.append(stock_df)
        df = pd.concat(returns, ignore_index=True)
        
        # Lag 피처 추가
        df = self.add_lag_features(df)
        
        print(f"   데이터: {len(df):,}건")
        return df
    
    def add_lag_features(self, df):
        """Lag 피처 추가"""
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
        """피처 준비"""
        exclude_cols = ['날짜', 'target', 'stock_code', 'stock_name', 'next_return']
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
    
    def backtest(self, target_date):
        """백테스트 실행"""
        print("=" * 70)
        print(f"📊 백테스트: {target_date} 예측 → 다음날 실제 결과")
        print("=" * 70)
        
        # 모델 로드
        self.load_models()
        
        # 데이터 로드
        df = self.load_data()
        
        # 타겟 날짜 데이터
        target_dt = pd.to_datetime(target_date)
        df_target = df[df['날짜'] == target_dt].copy()
        
        if df_target.empty:
            print(f"❌ {target_date} 데이터가 없습니다.")
            return None
        
        # NaN 제거
        before_drop = len(df_target)
        df_target = df_target.dropna(subset=['next_return'])
        
        print(f"\n🎯 예측 기준일: {target_date}")
        print(f"   예측 대상: {len(df_target)}개 종목")
        
        if len(df_target) == 0:
            print(f"\n   ⚠️ {target_date}은 데이터의 마지막 날짜라 다음날 결과가 없습니다.")
            print(f"   💡 이전 날짜로 시도해보세요: python backtest.py --date 2026-01-29")
            return None
        
        # 피처 준비
        X = self.prepare_features(df_target)
        
        # 예측
        results = []
        for idx, row in df_target.iterrows():
            stock_code = row['stock_code']
            stock_name = row['stock_name']
            next_return = row['next_return']  # 실제 다음날 수익률
            
            X_single = X.loc[[idx]]
            
            # 앙상블 예측
            probas = []
            for name, model in self.models.items():
                proba = model.predict_proba(X_single)[0][1]
                probas.append(proba)
            
            avg_proba = np.mean(probas)
            prediction = 1 if avg_proba >= 0.5 else 0
            
            # 실제 결과 (1% 기준)
            if next_return >= self.target_threshold:
                actual = 1  # 실제 1% 이상 상승
            elif next_return <= -self.target_threshold:
                actual = 0  # 실제 1% 이상 하락
            else:
                actual = -1  # 중간 구간 (노이즈)
            
            # 적중 여부
            if actual == -1:
                hit = "⚪ 무효"  # 중간 구간은 평가 제외
                hit_flag = None
            elif prediction == actual:
                hit = "✅ 적중"
                hit_flag = True
            else:
                hit = "❌ 실패"
                hit_flag = False
            
            results.append({
                'stock_code': stock_code,
                'stock_name': stock_name,
                'prediction': prediction,
                'probability': avg_proba,
                'actual_return': next_return,
                'actual': actual,
                'hit': hit,
                'hit_flag': hit_flag
            })
        
        results_df = pd.DataFrame(results)
        
        # 결과 출력
        self.display_results(results_df)
        
        return results_df
    
    def display_results(self, results_df):
        """결과 출력"""
        if results_df is None or results_df.empty:
            print("\n❌ 표시할 결과가 없습니다.")
            return
            
        print("\n" + "=" * 70)
        print("📈 종목별 예측 결과")
        print("=" * 70)
        
        # 상승 예측 종목
        up_preds = results_df[results_df['prediction'] == 1].sort_values('probability', ascending=False)
        down_preds = results_df[results_df['prediction'] == 0].sort_values('probability')
        
        print(f"\n🔺 상승 예측 ({len(up_preds)}개)")
        print("-" * 70)
        print(f"{'종목명':12s} {'코드':8s} {'예측확률':>8s} {'실제수익률':>10s} {'결과':>8s}")
        print("-" * 70)
        for _, row in up_preds.iterrows():
            print(f"{str(row['stock_name']):12s} {str(row['stock_code']).zfill(6):8s} "
                  f"{row['probability']*100:7.1f}% {row['actual_return']*100:9.2f}% {row['hit']:>8s}")
        
        print(f"\n🔻 하락 예측 ({len(down_preds)}개)")
        print("-" * 70)
        print(f"{'종목명':12s} {'코드':8s} {'예측확률':>8s} {'실제수익률':>10s} {'결과':>8s}")
        print("-" * 70)
        for _, row in down_preds.iterrows():
            print(f"{str(row['stock_name']):12s} {str(row['stock_code']).zfill(6):8s} "
                  f"{(1-row['probability'])*100:7.1f}% {row['actual_return']*100:9.2f}% {row['hit']:>8s}")
        
        # 통계
        print("\n" + "=" * 70)
        print("📋 적중률 통계")
        print("=" * 70)
        
        # 유효한 결과만 (중간 구간 제외)
        valid_results = results_df[results_df['hit_flag'].notna()]
        
        if len(valid_results) > 0:
            total_valid = len(valid_results)
            total_hit = valid_results['hit_flag'].sum()
            hit_rate = total_hit / total_valid * 100
            
            print(f"   총 예측: {len(results_df)}개")
            print(f"   유효 결과 (±1% 이상): {total_valid}개")
            print(f"   무효 결과 (노이즈 구간): {len(results_df) - total_valid}개")
            print(f"\n   ✅ 적중: {total_hit}개")
            print(f"   ❌ 실패: {total_valid - total_hit}개")
            print(f"\n   🎯 적중률: {hit_rate:.1f}%")
            
            # 상승/하락 별 적중률
            up_valid = valid_results[valid_results['prediction'] == 1]
            down_valid = valid_results[valid_results['prediction'] == 0]
            
            if len(up_valid) > 0:
                up_hit_rate = up_valid['hit_flag'].sum() / len(up_valid) * 100
                print(f"   🔺 상승 예측 적중률: {up_hit_rate:.1f}% ({int(up_valid['hit_flag'].sum())}/{len(up_valid)})")
            
            if len(down_valid) > 0:
                down_hit_rate = down_valid['hit_flag'].sum() / len(down_valid) * 100
                print(f"   🔻 하락 예측 적중률: {down_hit_rate:.1f}% ({int(down_valid['hit_flag'].sum())}/{len(down_valid)})")
        else:
            print("   유효한 결과가 없습니다.")
        
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='예측 결과 백테스트')
    parser.add_argument('--date', '-d', type=str, required=True,
                        help='백테스트 날짜 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='결과 저장 경로 (CSV)')
    
    args = parser.parse_args()
    
    backtester = Backtester()
    results = backtester.backtest(args.date)
    
    if args.output and results is not None:
        results.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"\n💾 저장: {args.output}")


if __name__ == "__main__":
    main()

