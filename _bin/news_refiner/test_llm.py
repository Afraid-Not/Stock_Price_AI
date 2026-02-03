"""
LLM 뉴스 분석 테스트 (최근 100건만)
"""
import pandas as pd
import os
import sys

# .env 파일 찾기 (여러 위치에서)
try:
    from dotenv import load_dotenv
    
    possible_paths = [
        'D:/stock/.env',  # 재현님 .env 위치
        '.env',
        '../.env',
        '../../.env',
        os.path.expanduser('~/.env'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            load_dotenv(path)
            print(f"✅ .env 로드: {path}")
            break
    else:
        load_dotenv()  # 기본 위치
except ImportError:
    print("⚠️ python-dotenv 없음")

# API 키 확인
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
    print("\n다음 중 하나를 실행하세요:")
    print("  1. PowerShell: $env:OPENAI_API_KEY='sk-...'")
    print("  2. .env 파일에 OPENAI_API_KEY=sk-... 추가")
    sys.exit(1)
else:
    print(f"✅ API 키 확인됨: {api_key[:10]}...")

# 뉴스 데이터 로드
print("\n📂 뉴스 데이터 로드...")
news_path = "_news/news_005930_20200101_20260127.csv"
df = pd.read_csv(news_path, encoding='utf-8-sig')
print(f"   총 {len(df)}건")

# 최근 100건만 선택
df_test = df.head(100).copy()
print(f"   테스트: 최근 100건")

# LLM 분석
from llm_refiner import LLMNewsRefiner

print("\n🤖 LLM 분석 시작...")
refiner = LLMNewsRefiner(provider="openai", model="gpt-4o-mini")

# 분석 실행
df_result = refiner.analyze_dataframe(
    df_test,
    title_column="HTS_공시_제목_내용",
    date_column="작성일자",
    stock_name="삼성전자",
    delay=0.3  # 빠른 테스트
)

# 결과 저장
output_path = "_news/test_llm_result.csv"
df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n💾 결과 저장: {output_path}")

# 날짜별 집계
df_result['날짜'] = df_result['작성일자'].astype(str).apply(
    lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if len(x) == 8 else x
)
df_daily = refiner.aggregate_daily(df_result, date_column='날짜')

daily_path = "_news/test_llm_daily.csv"
df_daily.to_csv(daily_path, index=False, encoding='utf-8-sig')
print(f"💾 일별 집계 저장: {daily_path}")

# 결과 미리보기
print("\n" + "=" * 60)
print("📊 분석 결과 샘플")
print("=" * 60)
sample_cols = ['날짜', 'HTS_공시_제목_내용', 'llm_sentiment', 'llm_impact', 'llm_event_type']
print(df_result[sample_cols].head(10).to_string())

print("\n" + "=" * 60)
print("📅 일별 집계 결과")
print("=" * 60)
print(df_daily.to_string())

