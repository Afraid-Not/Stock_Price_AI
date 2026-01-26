import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler

class KISPreprocessor:
    def __init__(self):
        # s05_preprocessing.py의 selected_columns 그대로 유지
        self.selected_columns = [
            '날짜', '종가', '시가', '고가', '저가', '거래량', 
            '거래대금', '등락률', '외국인_순매수금액', '기관계_순매수금액', 
            '개인_순매수금액', '금융투자_순매수금액', '투신_순매수금액', 
            '사모펀드_순매수금액', '은행_순매수금액', '보험_순매수금액', 
            '연기금_순매수금액', '기타금융_순매수금액', '기타법인_순매수금액', 
            'frgn_shnu_tr_pbmn', 'frgn_seln_tr_pbmn', 
            'orgn_shnu_tr_pbmn', 'orgn_seln_tr_pbmn', 'prsn_shnu_tr_pbmn', 
            'prsn_seln_tr_pbmn', '외국인_순매수수량', '기관계_순매수수량', '개인_순매수수량'
        ]
        # s05_preprocessing.py의 클리핑 대상 리스트
        self.cols_to_clip = [
            '외국인_순매수금액', '기관계_순매수금액', '개인_순매수금액', 
            '금융투자_순매수금액', '투신_순매수금액', '사모펀드_순매수금액', 
            '은행_순매수금액', '보험_순매수금액', '연기금_순매수금액', 
            '기타금융_순매수금액', '외국인_순매수수량', '기관계_순매수수량', '개인_순매수수량'
        ]

    def run(self, df):
        # 1. 컬럼 추출 및 기본 처리
        ss_df = df[self.selected_columns].copy()
        ss_df['거래량'] = ss_df['거래량'].replace(0, np.nan)
        ss_df = ss_df.dropna(axis=0)

        # 2. 가격 로그 수익률 변환
        # $r_t = \ln(P_t / P_{t-1})$
        price_cols = ['종가', '시가', '고가', '저가']
        for col in price_cols:
            ss_df[f'{col}_log_ret'] = np.log(ss_df[col].astype(float) / ss_df[col].astype(float).shift(1))

        # 3. 거래량/거래대금 로그 변환
        ss_df['거래량'] = np.log1p(ss_df['거래량'].astype(float))
        ss_df['거래대금'] = np.log1p(ss_df['거래대금'].astype(float))

        # 4. 수급 데이터 RobustScaler (flow_cols 리스트 동일)
        flow_cols = [
            '외국인_순매수금액', '기관계_순매수금액', '개인_순매수금액', '금융투자_순매수금액', 
            '투신_순매수금액', '사모펀드_순매수금액', '은행_순매수금액', '보험_순매수금액', 
            '연기금_순매수금액', '기타금융_순매수금액', '기타법인_순매수금액', 
            'frgn_shnu_tr_pbmn', 'frgn_seln_tr_pbmn', 'orgn_shnu_tr_pbmn', 
            'orgn_seln_tr_pbmn', 'prsn_shnu_tr_pbmn', 'prsn_seln_tr_pbmn', 
            '외국인_순매수수량', '기관계_순매수수량', '개인_순매수수량'
        ]
        ss_df[flow_cols] = RobustScaler().fit_transform(ss_df[flow_cols])

        # 5. 등락률 StandardScaler
        ss_df['등락률'] = StandardScaler().fit_transform(ss_df[['등락률']])
        ss_df = ss_df.dropna().reset_index(drop=True)

        # 6. 이동평균선 및 이격도
        ss_df['MA5'] = ss_df['종가'].rolling(window=5).mean()
        ss_df['MA10'] = ss_df['종가'].rolling(window=10).mean()
        ss_df['disparity_5'] = (ss_df['종가'] / ss_df['MA5']) - 1
        ss_df['disparity_10'] = (ss_df['종가'] / ss_df['MA10']) - 1
        ss_df['ma_gap'] = (ss_df['MA5'] / ss_df['MA10']) - 1
        ss_df['ma5_gradient'] = np.log(ss_df['MA5'] / ss_df['MA5'].shift(1))
        
        ss_df = ss_df.drop(columns=['MA5', 'MA10']).dropna().reset_index(drop=True)

        # 7. 시간 주기적 인코딩 (수치 4, 12, 31 엄수)
        ss_df['날짜'] = pd.to_datetime(ss_df['날짜'])
        day_of_week = ss_df['날짜'].dt.dayofweek
        month = ss_df['날짜'].dt.month
        day = ss_df['날짜'].dt.day

        ss_df['day_sin'] = np.sin(2 * np.pi * day_of_week / 4)
        ss_df['day_cos'] = np.cos(2 * np.pi * day_of_week / 4)
        ss_df['month_sin'] = np.sin(2 * np.pi * month / 12)
        ss_df['month_cos'] = np.cos(2 * np.pi * month / 12)
        ss_df['day_month_sin'] = np.sin(2 * np.pi * day / 31)
        ss_df['day_month_cos'] = np.cos(2 * np.pi * day / 31)

        # 8. 하드 클리핑 및 분위수 클리핑
        ss_df['은행_순매수금액'] = ss_df['은행_순매수금액'].clip(-24, 20)
        ss_df['보험_순매수금액'] = ss_df['보험_순매수금액'].clip(-10, 10)

        for col in self.cols_to_clip:
            lower = ss_df[col].quantile(0.01)
            upper = ss_df[col].quantile(0.99)
            ss_df[col] = ss_df[col].clip(lower, upper)

        # 9. 타겟 생성 및 불필요 컬럼 제거
        ss_df['target'] = ss_df['종가_log_ret'].shift(-1)
        ss_df = ss_df.dropna().reset_index(drop=True)
        # 종가, 시가, 고가, 저가, 종가_log_ret 삭제
        ss_df = ss_df.drop(['종가', '시가', '고가', '저가', '종가_log_ret'], axis=1)
        
        return ss_df