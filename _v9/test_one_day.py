"""하루치 뉴스 테스트 분석"""
import pandas as pd
from n02_analyze_news import NewsAnalyzer

# 삼성전자 뉴스 로드
df = pd.read_csv('_data/news/news_005930_20250203_20260203.csv', encoding='utf-8-sig')
print(f'총 뉴스: {len(df)}건')

# 최근 하루치만 추출
df['date'] = df['작성일자'].astype(str)
latest_date = df['date'].max()
df_one_day = df[df['date'] == latest_date].copy()
df_one_day['stock_code'] = '005930'
print(f'최근 날짜: {latest_date}')
print(f'하루 뉴스: {len(df_one_day)}건')

# LLM 분석
print('\n🤖 LLM 분석 시작...')
analyzer = NewsAnalyzer(method='llm')
df_result = analyzer.analyze_dataframe(df_one_day, delay=0.3)

print('\n' + '='*60)
print('📊 분석 결과 샘플')
print('='*60)
for _, row in df_result.head(15).iterrows():
    title = row['HTS_공시_제목_내용'][:50] + '...' if len(str(row['HTS_공시_제목_내용'])) > 50 else row['HTS_공시_제목_내용']
    print(f"[{row['sentiment']:+.2f}] (영향:{row['impact']}) {row['event_type']:4s} | {title}")

print('\n📈 통계')
print(f"  평균 감성: {df_result['sentiment'].mean():.3f}")
print(f"  긍정 뉴스: {len(df_result[df_result['sentiment'] > 0.2])}건")
print(f"  부정 뉴스: {len(df_result[df_result['sentiment'] < -0.2])}건")

