import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

class StockPreprocessor:
    def __init__(self, scaler_dir="D:/stock/_v6/scalers", stock_code=None):
        self.scaler_dir = scaler_dir
        self.stock_code = stock_code or "DEFAULT"
        os.makedirs(self.scaler_dir, exist_ok=True)
        
        # RobustScaler: 이상치에 강함 (수급 데이터용)
        self.flow_scaler = RobustScaler()
        # StandardScaler: 기술적 지표용
        self.indicator_scaler = StandardScaler()
        # MinMaxScaler: 수익률 컬럼 및 target을 -0.3 ~ 0.3 범위로 정규화
        self.return_scaler = MinMaxScaler(feature_range=(-0.3, 0.3))
        self.target_scaler = MinMaxScaler(feature_range=(-0.3, 0.3))
        
        # 1. 핵심 수급 컬럼 (주요 3주체에 집중하여 노이즈 제거)
        self.main_flow_cols = [
            '외국인_순매수금액', '기관계_순매수금액', '개인_순매수금액',
            '외국인_순매수수량', '기관계_순매수수량', '개인_순매수수량'
        ]
        
        # 2. 기술적/가격 지표 컬럼 (스케일링 적용)
        self.indicator_cols = [
            'disparity_5', 'disparity_10', 'ma_gap', 'ma5_gradient', 'volatility_5'
        ]
        
        # 3. 수익률 컬럼 (-0.3 ~ 0.3 범위로 정규화)
        # 종가_ret은 target으로 사용하므로 피처에서 제외
        self.return_cols = [
            '시가_ret', '고가_ret', '저가_ret'
        ]

    def run_pipeline(self, input_path, output_path, is_train=True):
        df = pd.read_csv(input_path)
        
        # 날짜 정렬 (날짜 외의 NaN은 여기서 지우지 않음)
        df['날짜'] = pd.to_datetime(df['날짜'].astype(str), format='%Y%m%d')
        df = df.sort_values('날짜').reset_index(drop=True)

        # 0. 종가 관련 피쳐들 클리핑 (하위 1%, 상위 1%)
        # 종가에 영향을 받는 피쳐들이 있으므로 먼저 클리핑
        # 원본 가격을 클리핑하면 _ret 계산 시에도 극단값이 자연스럽게 완화됨
        price_cols = ['시가', '고가', '저가', '종가']
        for col in price_cols:
            if col in df.columns:
                lower_bound = df[col].quantile(0.01)
                upper_bound = df[col].quantile(0.99)
                # FutureWarning 방지를 위해 원본 dtype 유지
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound).astype(df[col].dtype)
                print(f"   ✅ {col} clipped: [{lower_bound:.2f}, {upper_bound:.2f}]")

        # 1. 기술적 지표 생성 (이 함수가 호출되어야 종가_ret이 생김)
        # 원본 가격이 이미 클리핑되어 있으므로 _ret도 자연스럽게 극단값이 완화됨
        df = self._add_technical_indicators(df)
        
        # 2. 타겟 생성: 종가_ret을 target으로 사용 (전일 종가 대비 오늘 종가 수익률)
        df['target'] = df['종가_ret']
        
        # 3. 날짜 피처 인코딩
        df = self._add_date_features(df)
        
        # 4. 거래량/대금 로그 변환
        df['거래량'] = np.log1p(df['거래량'].replace(0, np.nan))
        df['거래대금'] = np.log1p(df['거래대금'].replace(0, np.nan))

        # 5. [중요] 필수 학습 컬럼에 대해서만 결측치 제거
        # 전체 113개 컬럼 중 학습에 쓸 핵심 컬럼들에 데이터가 있는 행만 남깁니다.
        required_cols = self.main_flow_cols + self.indicator_cols + self.return_cols + ['target', '거래량', '거래대금']
        df = df.dropna(subset=required_cols).reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("⚠️ 전처리 후 남은 데이터가 0건입니다. 데이터 수집 기간을 더 늘려주세요 (최소 30일 이상 권장).")

        # 6. 스케일링 적용 (수익률 컬럼도 -0.3 ~ 0.3 범위로 정규화 포함)
        df = self._apply_scaling(df, is_train)
        
        # 7. 최종 데이터셋 정리 (학습에 쓰지 않을 원본 컬럼 제거)
        final_df = self._prepare_final_dataset(df)
        
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 전처리 완료: {output_path} (남은 데이터: {len(final_df)}행, 피처 수: {len(final_df.columns)-1})")
        return final_df

    def _add_technical_indicators(self, df):
        # 수익률 (전일 종가 대비)
        # 시가/고가/저가/종가는 전일 종가와 비교
        전일종가 = df['종가'].shift(1)
        for col in ['시가', '고가', '저가', '종가']:
            df[f'{col}_ret'] = (df[col] - 전일종가) / 전일종가
        
        # 이동평균 및 이격도
        ma5 = df['종가'].rolling(window=5).mean()
        ma10 = df['종가'].rolling(window=10).mean()
        
        df['disparity_5'] = (df['종가'] / ma5) - 1
        df['disparity_10'] = (df['종가'] / ma10) - 1
        df['ma_gap'] = (ma5 / ma10) - 1
        df['ma5_gradient'] = (ma5 - ma5.shift(1)) / ma5.shift(1)  # 이동평균의 변화율
        
        # 변동성 지표 (최근 5일 등락률 표준편차)
        df['volatility_5'] = df['등락률'].rolling(window=5).std()
        
        return df

    def _apply_scaling(self, df, is_train):
        # 종목별 스케일러 파일 경로
        flow_path = os.path.join(self.scaler_dir, f"{self.stock_code}_flow_scaler.bin")
        ind_path = os.path.join(self.scaler_dir, f"{self.stock_code}_ind_scaler.bin")
        return_path = os.path.join(self.scaler_dir, f"{self.stock_code}_return_scaler.bin")
        target_path = os.path.join(self.scaler_dir, f"{self.stock_code}_target_scaler.bin")

        if is_train:
            df[self.main_flow_cols] = self.flow_scaler.fit_transform(df[self.main_flow_cols])
            df[self.indicator_cols] = self.indicator_scaler.fit_transform(df[self.indicator_cols])
            # 수익률 컬럼을 -0.3 ~ 0.3 범위로 정규화
            df[self.return_cols] = self.return_scaler.fit_transform(df[self.return_cols])
            # target도 -0.3 ~ 0.3 범위로 정규화
            df[['target']] = self.target_scaler.fit_transform(df[['target']])
            joblib.dump(self.flow_scaler, flow_path)
            joblib.dump(self.indicator_scaler, ind_path)
            joblib.dump(self.return_scaler, return_path)
            joblib.dump(self.target_scaler, target_path)
        else:
            self.flow_scaler = joblib.load(flow_path)
            self.indicator_scaler = joblib.load(ind_path)
            self.return_scaler = joblib.load(return_path)
            self.target_scaler = joblib.load(target_path)
            df[self.main_flow_cols] = self.flow_scaler.transform(df[self.main_flow_cols])
            df[self.indicator_cols] = self.indicator_scaler.transform(df[self.indicator_cols])
            df[self.return_cols] = self.return_scaler.transform(df[self.return_cols])
            df[['target']] = self.target_scaler.transform(df[['target']])
        
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
            self.return_cols +  # 수익률 컬럼 (스케일링 미적용)
            self.indicator_cols + 
            ['day_sin', 'day_cos', 'month_sin', 'month_cos']
        )
        # 존재하는 컬럼만 선택
        available_cols = [c for c in final_features if c in df.columns]
        return df[available_cols].reset_index(drop=True)