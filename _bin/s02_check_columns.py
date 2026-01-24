import pandas as pd

# 1. 데이터 로드 (이전 단계에서 정리된 데이터를 가정합니다)
file_path = "D:/stock/_data/stock/stock_005930_20220101_20251231.csv"
ss = pd.read_csv(file_path)

# 2. 전처리 (컬럼명 변경 및 날짜 형식 변환)
rename_dict = {
    '일자': 'date', '종가': 'final', '고가': 'high', '저가': 'low', 
    '개인순매수': 'personal', '외인순매수': 'foreigner', '기관순매수': 'institute'
}
ss = ss.rename(columns=rename_dict)
if 'PBR' in ss.columns:
    ss = ss.drop('PBR', axis=1)

ss['date'] = pd.to_datetime(ss['date'], format='%Y%m%d')
ss = ss.sort_values('date').reset_index(drop=True)

# 3. 전날 대비 변동폭(Difference) 컬럼 생성
# .diff()는 현재 행의 값에서 바로 이전 행의 값을 뺀 결과를 돌려줍니다.

# (1) 가격 변동폭 및 변동률
ss['diff_final'] = ss['final'].diff()               # 가격 변동액 (원)
ss['pct_final'] = ss['final'].pct_change() * 100    # 가격 변동률 (%)

# (2) 수급 변동폭 (어제보다 얼마나 더/덜 샀는지)
ss['diff_personal'] = ss['personal'].diff()
ss['diff_foreigner'] = ss['foreigner'].diff()
ss['diff_institute'] = ss['institute'].diff()

# 1. 고가-저가 절대 간극 (원 단위)
ss['range'] = ss['high'] - ss['low']
ss['range_pct'] = (ss['range'] / ss['final']) * 100
ss = ss.dropna()

# 3. 결과 확인
print("--- 변동성 지표 추가 결과 ---")
print(ss.head())

# 6. 저장 (변동폭이 포함된 새로운 파일)
# save_path = "D:/stock/_data/stock/stock_analysis_005930.csv"
# ss.to_csv(save_path, index=False, encoding='utf-8-sig')