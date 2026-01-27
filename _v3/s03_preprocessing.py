import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import RobustScaler, StandardScaler

class StockPreprocessor:
    def __init__(self, scaler_dir="D:/stock/_v3/scalers"):
        self.scaler_dir = scaler_dir
        os.makedirs(self.scaler_dir, exist_ok=True)
        
        # 스케일러 인스턴스 초기화
        self.flow_scaler = RobustScaler()
        self.return_scaler = StandardScaler()
        
        # 컬럼 정의
        self.selected_columns = [
            '날짜', '종가', '시가', '고가', '저가', '거래량', 
            '거래대금', '등락률', '외국인_순매수금액', '기관계_순매수금액', 
            '개인_순매수금액', '금융투자_순매수금액', '투신_순매수금액', 
            '사모펀드_순매수금액', '은행_순매수금액', '보험_순매수금액', 
            '연기금_순매수금액', '기타금융_순매수금액', '기타법인_순매수금액', 
            '외국인_매수금액', '외국인_매도금액', '기관계_매수금액', '기관계_매도금액', 
            '개인_매수금액', '개인_매도금액', '외국인_순매수수량', '기관계_순매수수량', '개인_순매수수량'
        ]
        
        self.flow_cols = [
            '외국인_순매수금액', '기관계_순매수금액', '개인_순매수금액', '금융투자_순매수금액', 
            '투신_순매수금액', '사모펀드_순매수금액', '은행_순매수금액', '보험_순매수금액', 
            '연기금_순매수금액', '기타금융_순매수금액', '기타법인_순매수금액', 
            '외국인_매수금액', '외국인_매도금액', '기관계_매수금액', '기관계_매도금액', 
            '개인_매수금액', '개인_매도금액', '외국인_순매수수량', '기관계_순매수수량', '개인_순매수수량'
        ]

    def run_pipeline(self, input_path, output_path, is_train=True):
        """is_train=True면 스케일러를 새로 학습시켜 저장하고, False면 저장된 것을 불러옵니다."""
        df = pd.read_csv(input_path)
        ss_df = df[self.selected_columns].copy()

        ss_df['거래량'] = ss_df['거래량'].replace(0, np.nan)
        ss_df = ss_df.dropna(axis=0)

        # 1. 기술적 지표
        ss_df = self._add_technical_indicators(ss_df)
        
        # 2. 스케일링 (학습/로드 분기)
        ss_df = self._apply_scaling(ss_df, is_train)
        
        # 3. 나머지 처리
        ss_df = self._add_date_features(ss_df)
        ss_df = self._apply_clipping(ss_df)
        ss_df = self._prepare_final_dataset(ss_df)

        ss_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        return ss_df

    def _apply_scaling(self, df, is_train):
        df['거래량'] = np.log1p(df['거래량'])
        df['거래대금'] = np.log1p(df['거래대금'])

        flow_scaler_path = os.path.join(self.scaler_dir, "flow_scaler.bin")
        return_scaler_path = os.path.join(self.scaler_dir, "return_scaler.bin")

        if is_train:
            # 학습 모드: fit_transform 후 저장
            df[self.flow_cols] = self.flow_scaler.fit_transform(df[self.flow_cols])
            df['등락률'] = self.return_scaler.fit_transform(df[['등락률']])
            
            joblib.dump(self.flow_scaler, flow_scaler_path)
            joblib.dump(self.return_scaler, return_scaler_path)
            print(f"💾 스케일러 저장 완료: {self.scaler_dir}")
        else:
            # 추론 모드: load 후 transform만
            self.flow_scaler = joblib.load(flow_scaler_path)
            self.return_scaler = joblib.load(return_scaler_path)
            
            df[self.flow_cols] = self.flow_scaler.transform(df[self.flow_cols])
            df['등락률'] = self.return_scaler.transform(df[['등락률']])
            print("🔌 기존 스케일러 로드 및 적용 완료")

        return df

    # (이하 _add_technical_indicators, _add_date_features, _apply_clipping, _prepare_final_dataset는 기존과 동일)
    def _add_technical_indicators(self, df):
        for col in ['종가', '시가', '고가', '저가']:
            df[f'{col}_log_ret'] = np.log(df[col] / df[col].shift(1))
        df['MA5'] = df['종가'].rolling(window=5).mean()
        df['MA10'] = df['종가'].rolling(window=10).mean()
        df['disparity_5'] = (df['종가'] / df['MA5']) - 1
        df['disparity_10'] = (df['종가'] / df['MA10']) - 1
        df['ma_gap'] = (df['MA5'] / df['MA10']) - 1
        df['ma5_gradient'] = np.log(df['MA5'] / df['MA5'].shift(1))
        return df.dropna().reset_index(drop=True)

    def _add_date_features(self, df):
        df['날짜'] = pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d')
        dw, m, d = df['날짜'].dt.dayofweek, df['날짜'].dt.month, df['날짜'].dt.day
        df['day_sin'] = np.sin(2 * np.pi * dw / 4)
        df['day_cos'] = np.cos(2 * np.pi * dw / 4)
        df['month_sin'] = np.sin(2 * np.pi * m / 12)
        df['month_cos'] = np.cos(2 * np.pi * m / 12)
        df['day_month_sin'] = np.sin(2 * np.pi * d / 31)
        df['day_month_cos'] = np.cos(2 * np.pi * d / 31)
        return df

    def _apply_clipping(self, df):
        for col in self.flow_cols:
            df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))
        return df

    def _prepare_final_dataset(self, df):
        df['target'] = df['종가_log_ret'].shift(-1)
        df = df.dropna().reset_index(drop=True)
        return df.drop(columns=['종가', '시가', '고가', '저가', '종가_log_ret', 'MA5', 'MA10'])