from pykrx import stock

# 날짜를 오늘보다 확실히 전인 날짜(예: 2025년 12월 26일)로 테스트해보세요.
target_date = "20251226" 
df = stock.get_market_ohlcv_by_ticker(target_date, market="KOSPI")

if df.empty:
    print(f"[{target_date}] 데이터가 비어 있습니다. 날짜를 확인하거나 잠시 후 시도하세요.")
else:
    print(df.head())