"""
9개 종목 뉴스 전체 LLM 감성 분석
"""
import os
import glob
import pandas as pd
import argparse
from n02_analyze_news import NewsAnalyzer

def main():
    parser = argparse.ArgumentParser(description="전체 뉴스 감성 분석")
    parser.add_argument("--input_dir", type=str, default="_data/news", help="뉴스 폴더")
    parser.add_argument("--output_dir", type=str, default="_data/news_sentiment", help="결과 폴더")
    parser.add_argument("--method", type=str, default="llm", choices=["llm", "finbert"])
    parser.add_argument("--delay", type=float, default=0.3, help="API 호출 간격")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 뉴스 파일 목록
    news_files = sorted(glob.glob(f"{args.input_dir}/news_*.csv"))
    
    print("=" * 60)
    print("🚀 전체 뉴스 감성 분석")
    print("=" * 60)
    print(f"입력 폴더: {args.input_dir}")
    print(f"출력 폴더: {args.output_dir}")
    print(f"분석 방법: {args.method}")
    print(f"파일 수: {len(news_files)}개")
    print("=" * 60)
    
    # 분석기 초기화 (한 번만)
    print("\n🔧 분석기 초기화 중...")
    analyzer = NewsAnalyzer(method=args.method)
    
    # 전체 통계
    total_news = 0
    total_processed = 0
    all_daily = []
    
    for idx, news_file in enumerate(news_files):
        filename = os.path.basename(news_file)
        stock_code = filename.split("_")[1]  # news_005930_... -> 005930
        
        # 이미 처리된 파일 스킵
        output_file = f"{args.output_dir}/sentiment_{stock_code}.csv"
        if os.path.exists(output_file):
            print(f"\n[{idx+1}/{len(news_files)}] {stock_code} - ⏭️ 이미 완료, 스킵")
            # 이미 완료된 파일도 daily 집계에 포함
            daily_file = f"{args.output_dir}/daily_{stock_code}.csv"
            if os.path.exists(daily_file):
                all_daily.append(pd.read_csv(daily_file, encoding='utf-8-sig'))
            continue
        
        print(f"\n[{idx+1}/{len(news_files)}] {stock_code} 분석 중...")
        
        try:
            # 뉴스 로드
            df = pd.read_csv(news_file, encoding='utf-8-sig')
            df['stock_code'] = stock_code
            total_news += len(df)
            print(f"   뉴스 수: {len(df)}건")
            
            # 감성 분석
            df_analyzed = analyzer.analyze_dataframe(df, delay=args.delay)
            
            if df_analyzed is not None and len(df_analyzed) > 0:
                total_processed += len(df_analyzed)
                
                # 분석 결과 저장
                df_analyzed.to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"   💾 분석 결과 저장: {output_file}")
                
                # 일별 집계
                df_daily = analyzer.aggregate_daily(df_analyzed)
                daily_output = f"{args.output_dir}/daily_{stock_code}.csv"
                df_daily.to_csv(daily_output, index=False, encoding='utf-8-sig')
                print(f"   💾 일별 집계 저장: {daily_output}")
                
                all_daily.append(df_daily)
            else:
                print(f"   ⚠️ 분석 실패")
                
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            import traceback
            traceback.print_exc()
    
    # 전체 daily 병합
    if all_daily:
        df_all_daily = pd.concat(all_daily, ignore_index=True)
        all_daily_path = f"{args.output_dir}/all_daily_sentiment.csv"
        df_all_daily.to_csv(all_daily_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 전체 일별 감성 저장: {all_daily_path}")
    
    print("\n" + "=" * 60)
    print("🎉 전체 분석 완료!")
    print(f"총 뉴스: {total_news}건")
    print(f"분석 완료: {total_processed}건")
    print("=" * 60)


if __name__ == "__main__":
    main()

