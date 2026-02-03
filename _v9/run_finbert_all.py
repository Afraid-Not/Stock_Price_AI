# -*- coding: utf-8 -*-
"""
모든 뉴스 파일에 대해 FinBert 분석 실행
"""
import os
from pathlib import Path
from analyze_finbert import analyze_news_file

def main():
    news_dir = Path("_data/news")
    output_dir = "_data/news_sentiment_finbert"
    
    # 이미 분석된 파일 확인
    done_files = set()
    if Path(output_dir).exists():
        for f in Path(output_dir).glob("finbert_*.csv"):
            stock_code = f.stem.replace("finbert_", "")
            done_files.add(stock_code)
    
    # 뉴스 파일 목록
    news_files = sorted(news_dir.glob("news_*.csv"))
    
    print("="*60)
    print(f"📂 총 뉴스 파일: {len(news_files)}개")
    print(f"✅ 이미 완료: {len(done_files)}개 - {done_files}")
    print("="*60)
    
    for i, news_file in enumerate(news_files, 1):
        stock_code = news_file.stem.split('_')[1]
        
        if stock_code in done_files:
            print(f"\n[{i}/{len(news_files)}] {stock_code} - 이미 완료, 건너뜀")
            continue
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(news_files)}] {stock_code} 분석 시작")
        print("="*60)
        
        try:
            analyze_news_file(str(news_file), output_dir)
            print(f"✅ {stock_code} 완료!")
        except Exception as e:
            print(f"❌ {stock_code} 에러: {e}")
            continue
    
    print("\n" + "="*60)
    print("🎉 모든 분석 완료!")
    print("="*60)

if __name__ == "__main__":
    main()

