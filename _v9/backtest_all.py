"""
여러 날짜 백테스트 실행 및 결과 저장
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from backtest import Backtester


def run_multiple_backtest(start_date, end_date, output_dir='_backtest'):
    """여러 날짜에 대해 백테스트 실행"""
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 백테스터 초기화
    backtester = Backtester()
    backtester.load_models()
    df = backtester.load_data()
    
    # 가용 날짜 확인
    available_dates = sorted(df['날짜'].unique())
    
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # 범위 내 날짜 필터링 (마지막 날짜 제외 - next_return 없음)
    test_dates = [d for d in available_dates if start_dt <= d <= end_dt]
    test_dates = test_dates[:-1]  # 마지막 날짜 제외
    
    print("=" * 70)
    print(f"📊 다중 백테스트 실행")
    print(f"   기간: {start_date} ~ {end_date}")
    print(f"   테스트 날짜 수: {len(test_dates)}일")
    print("=" * 70)
    
    all_results = []
    daily_stats = []
    
    for i, test_date in enumerate(test_dates):
        date_str = pd.Timestamp(test_date).strftime('%Y-%m-%d')
        print(f"\n[{i+1}/{len(test_dates)}] {date_str} 백테스트 중...", end=" ")
        
        try:
            # 해당 날짜 데이터 추출
            df_target = df[df['날짜'] == test_date].copy()
            df_target = df_target.dropna(subset=['next_return'])
            
            if df_target.empty:
                print("❌ 데이터 없음")
                continue
            
            # 피처 준비
            X = backtester.prepare_features(df_target)
            
            # 예측
            results = []
            for idx, row in df_target.iterrows():
                stock_code = row['stock_code']
                stock_name = row['stock_name']
                next_return = row['next_return']
                
                X_single = X.loc[[idx]]
                
                probas = []
                for name, model in backtester.models.items():
                    proba = model.predict_proba(X_single)[0][1]
                    probas.append(proba)
                
                avg_proba = np.mean(probas)
                prediction = 1 if avg_proba >= 0.5 else 0
                
                # 실제 결과
                threshold = backtester.target_threshold
                if next_return >= threshold:
                    actual = 1
                elif next_return <= -threshold:
                    actual = 0
                else:
                    actual = -1
                
                # 적중 여부
                if actual == -1:
                    hit_flag = None
                elif prediction == actual:
                    hit_flag = True
                else:
                    hit_flag = False
                
                results.append({
                    'date': date_str,
                    'stock_code': str(stock_code).zfill(6),
                    'stock_name': stock_name,
                    'prediction': prediction,
                    'probability': avg_proba,
                    'actual_return': next_return,
                    'actual': actual,
                    'hit_flag': hit_flag
                })
            
            results_df = pd.DataFrame(results)
            all_results.append(results_df)
            
            # 일별 통계
            valid = results_df[results_df['hit_flag'].notna()]
            if len(valid) > 0:
                hit_rate = valid['hit_flag'].sum() / len(valid) * 100
                daily_stats.append({
                    'date': date_str,
                    'total': len(results_df),
                    'valid': len(valid),
                    'hit': int(valid['hit_flag'].sum()),
                    'miss': len(valid) - int(valid['hit_flag'].sum()),
                    'hit_rate': hit_rate
                })
                print(f"✅ 적중률: {hit_rate:.1f}% ({int(valid['hit_flag'].sum())}/{len(valid)})")
            else:
                print("⚪ 유효 결과 없음")
                
        except Exception as e:
            print(f"❌ 오류: {e}")
    
    # 결과 저장
    if all_results:
        # 전체 결과
        all_df = pd.concat(all_results, ignore_index=True)
        all_path = f'{output_dir}/backtest_all_results.csv'
        all_df.to_csv(all_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 전체 결과 저장: {all_path}")
        
        # 일별 통계
        stats_df = pd.DataFrame(daily_stats)
        stats_path = f'{output_dir}/backtest_daily_stats.csv'
        stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
        print(f"💾 일별 통계 저장: {stats_path}")
        
        # 요약 출력
        print("\n" + "=" * 70)
        print("📋 전체 백테스트 요약")
        print("=" * 70)
        
        total_valid = all_df[all_df['hit_flag'].notna()]
        if len(total_valid) > 0:
            total_hit = total_valid['hit_flag'].sum()
            overall_hit_rate = total_hit / len(total_valid) * 100
            
            print(f"   테스트 기간: {stats_df['date'].min()} ~ {stats_df['date'].max()}")
            print(f"   테스트 일수: {len(stats_df)}일")
            print(f"   총 예측 수: {len(all_df):,}건")
            print(f"   유효 결과: {len(total_valid):,}건")
            print(f"   적중: {int(total_hit):,}건")
            print(f"   실패: {len(total_valid) - int(total_hit):,}건")
            print(f"\n   🎯 전체 적중률: {overall_hit_rate:.1f}%")
            
            # 상승/하락별
            up_valid = total_valid[total_valid['prediction'] == 1]
            down_valid = total_valid[total_valid['prediction'] == 0]
            
            if len(up_valid) > 0:
                up_rate = up_valid['hit_flag'].sum() / len(up_valid) * 100
                print(f"   🔺 상승 예측 적중률: {up_rate:.1f}% ({int(up_valid['hit_flag'].sum())}/{len(up_valid)})")
            
            if len(down_valid) > 0:
                down_rate = down_valid['hit_flag'].sum() / len(down_valid) * 100
                print(f"   🔻 하락 예측 적중률: {down_rate:.1f}% ({int(down_valid['hit_flag'].sum())}/{len(down_valid)})")
            
            # 일별 평균
            avg_hit_rate = stats_df['hit_rate'].mean()
            print(f"\n   📈 일별 평균 적중률: {avg_hit_rate:.1f}%")
            print(f"   📉 최저 적중률: {stats_df['hit_rate'].min():.1f}%")
            print(f"   📈 최고 적중률: {stats_df['hit_rate'].max():.1f}%")
        
        print("=" * 70)
        
        return all_df, stats_df
    
    return None, None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='다중 백테스트')
    parser.add_argument('--start', '-s', type=str, default='2026-01-01',
                        help='시작일 (기본: 2026-01-01)')
    parser.add_argument('--end', '-e', type=str, default='2026-01-30',
                        help='종료일 (기본: 2026-01-30)')
    parser.add_argument('--output', '-o', type=str, default='_backtest',
                        help='결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    run_multiple_backtest(args.start, args.end, args.output)

