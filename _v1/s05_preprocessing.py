import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

# 1. 데이터 로드
df = pd.read_csv("D:/stock/_data/manual_fetch/005930_renamed.csv")

# 2. 필요한 컬럼 추출
selected_columns = [
    '날짜', '종가', '시가', '고가', '저가', '거래량', 
    '거래대금', '등락률', '외국인_순매수금액', '기관계_순매수금액', 
    '개인_순매수금액', '금융투자_순매수금액', '투신_순매수금액', 
    '사모펀드_순매수금액', '은행_순매수금액', '보험_순매수금액', 
    '연기금_순매수금액', '기타금융_순매수금액', '기타법인_순매수금액', 
    'frgn_shnu_tr_pbmn', 'frgn_seln_tr_pbmn', 
    'orgn_shnu_tr_pbmn', 'orgn_seln_tr_pbmn', 'prsn_shnu_tr_pbmn', 
    'prsn_seln_tr_pbmn', '외국인_순매수수량', '기관계_순매수수량', '개인_순매수수량'
]
ss_df = df[selected_columns].copy()

# 3. 데이터 전처리: 거래량이 0인 데이터를 결측치로 처리 후 제거
ss_df['거래량'] = ss_df['거래량'].replace(0, np.nan)
ss_df = ss_df.dropna(axis=0)

# 4. 결측치 확인
print("--- 결측치 확인 ---")
print(ss_df.isna().sum())
print("\n")

# 5. [추가] 각 컬럼별 통계량(Min, Max, Average) 출력
# '날짜' 컬럼은 수치가 아니므로 제외하고 계산합니다.
stats_df = ss_df.drop(columns=['날짜']).describe().T[['min', 'max', 'mean']]
stats_df.columns = ['Min', 'Max', 'Average']

print("--- 컬럼별 주요 통계량 (Min, Max, Average) ---")
print(stats_df)
print("\n")

# --- 특성을 살리는 전략적 정규화 시작 ---

# [방법 1] 가격 데이터: 로그 수익률로 변환 (추세와 변동성 보존)
price_cols = ['종가', '시가', '고가', '저가']
for col in price_cols:
    ss_df[f'{col}_log_ret'] = np.log(ss_df[col] / ss_df[col].shift(1))

# [방법 2] 거래량/거래대금: 로그 변환 (치우침 해소)
ss_df['거래량'] = np.log1p(ss_df['거래량'])
ss_df['거래대금'] = np.log1p(ss_df['거래대금'])

# [방법 3] 수급 데이터: RobustScaler (아웃라이어 영향 최소화)
# 매수/매도 금액 등 방향성이 중요한 데이터들
flow_cols = ['외국인_순매수금액', '기관계_순매수금액', 
    '개인_순매수금액', '금융투자_순매수금액', '투신_순매수금액', 
    '사모펀드_순매수금액', '은행_순매수금액', '보험_순매수금액', 
    '연기금_순매수금액', '기타금융_순매수금액', '기타법인_순매수금액', 
    'frgn_shnu_tr_pbmn', 'frgn_seln_tr_pbmn', 
    'orgn_shnu_tr_pbmn', 'orgn_seln_tr_pbmn', 'prsn_shnu_tr_pbmn', 
    'prsn_seln_tr_pbmn', '외국인_순매수수량', '기관계_순매수수량', '개인_순매수수량']
scaler = RobustScaler()
ss_df[flow_cols] = scaler.fit_transform(ss_df[flow_cols])

# [방법 4] 등락률: 이미 비율 데이터이므로 StandardScaler 정도만 적용
ss_df['등락률'] = StandardScaler().fit_transform(ss_df[['등락률']])

# 결측치 제거 (로그 수익률 계산으로 생긴 첫 행 NaN 제거)
ss_df = ss_df.dropna().reset_index(drop=True)

ss_df['MA5'] = ss_df['종가'].rolling(window=5).mean()
ss_df['MA10'] = ss_df['종가'].rolling(window=10).mean()

# 1. 이격도 계산 (별도 정규화 불필요, 이 자체로 훌륭한 피처)
ss_df['disparity_5'] = (ss_df['종가'] / ss_df['MA5']) - 1
ss_df['disparity_10'] = (ss_df['종가'] / ss_df['MA10']) - 1

ss_df['ma_gap'] = (ss_df['MA5'] / ss_df['MA10']) - 1

ss_df['ma5_gradient'] = np.log(ss_df['MA5'] / ss_df['MA5'].shift(1))

ss_df = ss_df.drop(columns=['MA5', 'MA10'])
ss_df = ss_df.dropna().reset_index(drop=True)

# 1. 날짜 타입을 datetime으로 변경
ss_df['날짜'] = pd.to_datetime(ss_df['날짜'])

ss_df['day_of_week'] = ss_df['날짜'].dt.dayofweek
ss_df['month'] = ss_df['날짜'].dt.month
ss_df['day'] = ss_df['날짜'].dt.day

ss_df['day_sin'] = np.sin(2 * np.pi * ss_df['day_of_week'] / 4)
ss_df['day_cos'] = np.cos(2 * np.pi * ss_df['day_of_week'] / 4)

# 월 변환 (1~12 범위이므로 max는 12)
ss_df['month_sin'] = np.sin(2 * np.pi * ss_df['month'] / 12)
ss_df['month_cos'] = np.cos(2 * np.pi * ss_df['month'] / 12)

# 주기적 인코딩 적용 (최대 일수를 31로 설정)
ss_df['day_month_sin'] = np.sin(2 * np.pi * ss_df['day'] / 31)
ss_df['day_month_cos'] = np.cos(2 * np.pi * ss_df['day'] / 31)

# 4. 사용한 기초 정보는 삭제 (선택 사항)
ss_df = ss_df.drop(columns=['day_of_week', 'month','day'])
ss_df['은행_순매수금액'] = ss_df['은행_순매수금액'].clip(-24, 20)
ss_df['보험_순매수금액'] = ss_df['보험_순매수금액'].clip(-10, 10)

# 1. 대상 컬럼 리스트 정의
cols_to_clip = [
    '외국인_순매수금액', '기관계_순매수금액', '개인_순매수금액', 
    '금융투자_순매수금액', '투신_순매수금액', '사모펀드_순매수금액', 
    '은행_순매수금액', '보험_순매수금액', '연기금_순매수금액', 
    '기타금융_순매수금액', '외국인_순매수수량', '기관계_순매수수량', '개인_순매수수량'
]

# 2. 분위수(Quantile) 기반 클리핑 적용
for col in cols_to_clip:
    # 상위 1%(0.99)와 하위 1%(0.01) 지점 계산
    lower_limit = ss_df[col].quantile(0.01)
    upper_limit = ss_df[col].quantile(0.99)
    
    # 해당 범위를 벗어나는 값을 임계값으로 고정
    ss_df[col] = ss_df[col].clip(lower_limit, upper_limit)

print(f"✅ {len(cols_to_clip)}개 수급 컬럼에 대해 1%/99% 클리핑을 완료했습니다.")

# 내일의 수익률을 오늘의 정답으로 설정
ss_df['target'] = ss_df['종가_log_ret'].shift(-1)
ss_df = ss_df.dropna().reset_index(drop=True)
ss_df = ss_df.drop(['종가', '시가', '고가', '저가', '종가_log_ret'], axis=1)
print(ss_df.columns)




print("--- 전처리 후 상위 5행 ---")
print(ss_df.head())

ss_df.to_csv("D:/stock/_data/manual_fetch/preprocessed_005930_20100101_20251231.csv", index=False)