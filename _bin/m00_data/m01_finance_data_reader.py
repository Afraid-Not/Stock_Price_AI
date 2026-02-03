import pandas as pd
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
from pykrx import stock

# ---------------------------------------------------------
# 1. 2024년 주가 데이터 수집 (삼성전자, SK하이닉스, 네이버)
# ---------------------------------------------------------
print("1. 주가 데이터 수집 중...")

target_stocks = {
    'Samsung': '005930',  # 삼성전자
    'SKHynix': '000660',  # SK하이닉스
    'Naver':   '035420'   # NAVER
}

stock_data = []

for name, code in target_stocks.items():
    # 2024년 1월 1일 ~ 2024년 12월 31일
    df = fdr.DataReader(code, '2024-01-01', '2024-12-31')
    df['Company'] = name  # 종목명 추가
    df['Change'] = df['Close'].pct_change() * 100  # 등락률 계산 (%)
    stock_data.append(df)

# 하나로 합치기
df_stock = pd.concat(stock_data)

# 날짜 인덱스를 컬럼으로 빼기 (Merge 할 때 편하려고)
df_stock = df_stock.reset_index() 

print(f"✅ 주가 수집 완료! 총 {len(df_stock)}행")
print(df_stock.head())

# ---------------------------------------------------------
# 2. 외인/기관 매매 정보 수집
# ---------------------------------------------------------
print("\n2. 외인/기관 매매 정보 수집 중...")

investor_data = []

for name, code in target_stocks.items():
    try:
        # 날짜별 외인/기관 매매 정보 가져오기 (날짜 형식: YYYYMMDD)
        df_investor = stock.get_market_trading_value_by_date(
            fromdate="20240101", 
            todate="20241231", 
            ticker=code
        )
        
        # 인덱스를 컬럼으로 변환 (날짜는 보통 인덱스로 되어있음)
        df_investor = df_investor.reset_index()
        
        # 날짜 컬럼 찾기 및 변환
        date_col = None
        for col in df_investor.columns:
            if col in ['날짜', 'date', 'Date', '일자']:
                date_col = col
                break
        
        # 첫 번째 컬럼이 날짜일 가능성
        if date_col is None and len(df_investor.columns) > 0:
            first_col = df_investor.columns[0]
            try:
                # 날짜 타입인지 확인
                test_date = pd.to_datetime(df_investor[first_col].iloc[0])
                date_col = first_col
            except:
                pass
        
        if date_col:
            df_investor['Date'] = pd.to_datetime(df_investor[date_col]).dt.date
        else:
            raise ValueError(f"날짜 컬럼을 찾을 수 없습니다. 컬럼: {df_investor.columns.tolist()}")
        
        df_investor['Company'] = name
        df_investor['StockCode'] = code
        
        # 첫 번째 종목의 경우 실제 컬럼명 출력 (디버깅)
        if len(investor_data) == 0:
            print(f"  디버깅 - 실제 컬럼명: {df_investor.columns.tolist()}")
        
        # 컬럼명 정리 (한글 -> 영문)
        # 실제 컬럼명 확인 후 매핑
        rename_dict = {}
        for col in df_investor.columns:
            col_str = str(col)
            # 매매대금 컬럼 (순매수/순매도가 포함된 경우와 아닌 경우 구분)
            if '기관합계' in col_str:
                if '순매수' in col_str or '순매도' in col_str or '순' in col_str:
                    rename_dict[col] = '기관합계순매수'
                else:
                    rename_dict[col] = '기관합계'
            elif '기타법인' in col_str:
                if '순매수' in col_str or '순매도' in col_str or '순' in col_str:
                    rename_dict[col] = '기타법인순매수'
                else:
                    rename_dict[col] = '기타법인'
            elif '개인' in col_str:
                if '순매수' in col_str or '순매도' in col_str or '순' in col_str:
                    rename_dict[col] = '개인순매수'
                else:
                    rename_dict[col] = '개인'
            elif '외국인' in col_str:
                if '순매수' in col_str or '순매도' in col_str or '순' in col_str:
                    rename_dict[col] = '외국인순매수'
                else:
                    rename_dict[col] = '외국인'
        
        df_investor = df_investor.rename(columns=rename_dict)
        
        investor_data.append(df_investor)
        print(f"  ✅ {name} ({code}) 외인/기관 정보 수집 완료")
        
    except Exception as e:
        print(f"  ⚠️ {name} ({code}) 외인/기관 정보 수집 실패: {e}")
        import traceback
        traceback.print_exc()

# 하나로 합치기
if investor_data:
    df_investor_all = pd.concat(investor_data, ignore_index=True)
    print(f"✅ 외인/기관 정보 수집 완료! 총 {len(df_investor_all)}행")
    print(f"실제 컬럼명: {df_investor_all.columns.tolist()}")
    print(df_investor_all.head())
    
    # 실제 존재하는 컬럼만 선택
    merge_cols = ['Date', 'Company']
    optional_cols = {
        'Foreign': ['외국인', 'Foreign'],
        'Foreign_Net': ['외국인순매수', 'Foreign_Net', '외국인 순매수'],
        'Institution': ['기관합계', 'Institution', '기관'],
        'Institution_Net': ['기관합계순매수', 'Institution_Net', '기관 순매수'],
        'Individual': ['개인', 'Individual'],
        'Individual_Net': ['개인순매수', 'Individual_Net', '개인 순매수']
    }
    
    for target_col, possible_names in optional_cols.items():
        for name in possible_names:
            if name in df_investor_all.columns:
                merge_cols.append(name)
                break
    
    print(f"병합에 사용할 컬럼: {merge_cols}")
    
    # 주가 데이터와 병합 (Date와 Company 기준)
    df_stock['Date'] = pd.to_datetime(df_stock['Date']).dt.date
    df_merged = pd.merge(
        df_stock, 
        df_investor_all[merge_cols],
        on=['Date', 'Company'],
        how='left'
    )
    
    print(f"\n✅ 데이터 병합 완료! 총 {len(df_merged)}행")
    print(df_merged.head())
    
    # 병합된 데이터 저장
    df_merged.to_csv("/home/jhkim/01_dev/03_stock_market_price_expectation/_data/02_stock/stock_20240101-20241231.csv", index=False)
    print(f"\n✅ 최종 데이터 저장 완료!")
else:
    # 외인/기관 정보 수집 실패 시 기존 주가 데이터만 저장
    df_stock.to_csv("/home/jhkim/01_dev/03_stock_market_price_expectation/_data/02_stock/stock_20240101-20241231.csv", index=False)
    print("⚠️ 외인/기관 정보 수집 실패로 주가 데이터만 저장되었습니다.")