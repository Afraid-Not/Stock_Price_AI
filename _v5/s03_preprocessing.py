import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import RobustScaler, StandardScaler

class StockPreprocessor:
    def __init__(self, scaler_dir="D:/stock/_v5/scalers"):
        self.scaler_dir = scaler_dir
        os.makedirs(self.scaler_dir, exist_ok=True)
        
        # RobustScaler: 이상치에 강함 (수급 데이터용)
        self.flow_scaler = RobustScaler()
        # StandardScaler: 기술적 지표용
        self.indicator_scaler = StandardScaler()
        
        # 1. 핵심 수급 컬럼 (주요 3주체에 집중하여 노이즈 제거)
        self.main_flow_cols = [
            '외국인_순매수금액', '기관계_순매수금액', '개인_순매수금액',
            '외국인_순매수수량', '기관계_순매수수량', '개인_순매수수량'
        ]
        
        # 2. 기술적/가격 지표 컬럼
        self.indicator_cols = [
            '시가_log_ret', '고가_log_ret', '저가_log_ret', 
            'disparity_5', 'disparity_10', 'ma_gap', 'ma5_gradient', 'volatility_5'
        ]

    def run_pipeline(self, input_path, output_path, is_train=True):
        df = pd.read_csv(input_path)
        
        # 날짜 정렬 (날짜 외의 NaN은 여기서 지우지 않음)
        df['날짜'] = pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d')
        df = df.sort_values('날짜').reset_index(drop=True)

        # 1. 타겟 생성 (내일 종가 상승 여부: 1/0)
        df['target'] = (df['종가'].shift(-1) > df['종가']).astype(int)
        
        # 2. 기술적 지표 생성 (이 함수가 호출되어야 indicator_cols가 생김)
        df = self._add_technical_indicators(df)
        
        # 3. 날짜 피처 인코딩
        df = self._add_date_features(df)
        
        # 4. 거래량/대금 로그 변환
        df['거래량'] = np.log1p(df['거래량'].replace(0, np.nan))
        df['거래대금'] = np.log1p(df['거래대금'].replace(0, np.nan))

        # 5. [중요] 필수 학습 컬럼에 대해서만 결측치 제거
        # 전체 113개 컬럼 중 학습에 쓸 핵심 컬럼들에 데이터가 있는 행만 남깁니다.
        required_cols = self.main_flow_cols + self.indicator_cols + ['target', '거래량', '거래대금']
        df = df.dropna(subset=required_cols).reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("⚠️ 전처리 후 남은 데이터가 0건입니다. 데이터 수집 기간을 더 늘려주세요 (최소 30일 이상 권장).")

        # 6. 스케일링 적용
        df = self._apply_scaling(df, is_train)
        
        # 7. 최종 데이터셋 정리 (학습에 쓰지 않을 원본 컬럼 제거)
        final_df = self._prepare_final_dataset(df)
        
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 전처리 완료: {output_path} (남은 데이터: {len(final_df)}행, 피처 수: {len(final_df.columns)-1})")
        return final_df

    def _add_technical_indicators(self, df):
        # 로그 수익률
        for col in ['시가', '고가', '저가']:
            df[f'{col}_log_ret'] = np.log(df[col] / df[col].shift(1))
        
        # 이동평균 및 이격도
        ma5 = df['종가'].rolling(window=5).mean()
        ma10 = df['종가'].rolling(window=10).mean()
        
        df['disparity_5'] = (df['종가'] / ma5) - 1
        df['disparity_10'] = (df['종가'] / ma10) - 1
        df['ma_gap'] = (ma5 / ma10) - 1
        df['ma5_gradient'] = np.log(ma5 / ma5.shift(1))
        
        # 변동성 지표 (최근 5일 등락률 표준편차)
        df['volatility_5'] = df['등락률'].rolling(window=5).std()
        
        return df

    def _apply_scaling(self, df, is_train):
        flow_path = os.path.join(self.scaler_dir, "flow_scaler.bin")
        ind_path = os.path.join(self.scaler_dir, "ind_scaler.bin")

        if is_train:
            df[self.main_flow_cols] = self.flow_scaler.fit_transform(df[self.main_flow_cols])
            df[self.indicator_cols] = self.indicator_scaler.fit_transform(df[self.indicator_cols])
            joblib.dump(self.flow_scaler, flow_path)
            joblib.dump(self.indicator_scaler, ind_path)
        else:
            self.flow_scaler = joblib.load(flow_path)
            self.indicator_scaler = joblib.load(ind_path)
            df[self.main_flow_cols] = self.flow_scaler.transform(df[self.main_flow_cols])
            df[self.indicator_cols] = self.indicator_scaler.transform(df[self.indicator_cols])
        
        return df

    def _add_date_features(self, df):
        dw = df['날짜'].dt.dayofweek
        # 요일 (0~4: 월~금) -> 총 5개 지점
        df['day_sin'] = np.sin(2 * np.pi * dw / 5)
        df['day_cos'] = np.cos(2 * np.pi * dw / 5)

        # 월 (1~12) -> 총 12개 지점
        # 1월을 0점으로 맞추기 위해 (m-1) 사용
        m = df['날짜'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * (m - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (m - 1) / 12)
        
        return df

    def _prepare_final_dataset(self, df):
        # 앙상블 모델 학습에 포함할 최종 피처 리스트
        final_features = (
            ['target', '거래량', '거래대금'] + 
            self.main_flow_cols + 
            self.indicator_cols + 
            ['day_sin', 'day_cos', 'month_sin', 'month_cos']
        )
        # 존재하는 컬럼만 선택
        available_cols = [c for c in final_features if c in df.columns]
        return df[available_cols].reset_index(drop=True)