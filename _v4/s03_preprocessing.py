import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import RobustScaler, StandardScaler

class StockPreprocessor:
    def __init__(self, scaler_dir="D:/stock/_v4/scalers"):
        self.scaler_dir = scaler_dir
        os.makedirs(self.scaler_dir, exist_ok=True)
        
        # 스케일러 인스턴스 초기화
        # RobustScaler 사용: 비율 데이터는 이미 정규화되어 있지만, 분산 차이 조정 및 이상치에 강함
        self.flow_scaler = RobustScaler()
        self.return_scaler = RobustScaler()  # StandardScaler -> RobustScaler
        self.price_scaler = RobustScaler()   # StandardScaler -> RobustScaler
        
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

        print(f"  원본 데이터: {len(ss_df)}행")
        
        ss_df['거래량'] = ss_df['거래량'].replace(0, np.nan)
        
        # 필수 컬럼만 체크하여 dropna (거래량이 0이어도 다른 데이터는 유효할 수 있음)
        essential_cols = ['종가', '시가', '고가', '저가', '거래대금']
        essential_cols = [col for col in essential_cols if col in ss_df.columns]
        
        if essential_cols:
            ss_df = ss_df.dropna(subset=essential_cols)
            print(f"  필수 컬럼 체크 후: {len(ss_df)}행")
            
            if len(ss_df) == 0:
                print("  ⚠️ 경고: 필수 컬럼에 NaN이 많아 모든 행이 제거되었습니다.")
                return pd.DataFrame()
        else:
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
        # 거래량/대금 로그 변환
        df['거래량'] = np.log1p(df['거래량'])
        df['거래대금'] = np.log1p(df['거래대금'])

        # 절대 가격은 제거하고 상대적 지표만 사용
        # 가격 관련 상대적 지표들 (이미 비율/수익률 형태)
        price_relative_cols = [
            '종가_log_ret', '시가_log_ret', '고가_log_ret', '저가_log_ret',
            'high_low_ratio', 'open_close_ratio', 'high_close_ratio', 'low_close_ratio',
            'disparity_5', 'disparity_10', 'disparity_20',
            'ma_gap_5_10', 'ma_gap_10_20',
            'ma5_gradient', 'ma10_gradient',
            'momentum_3', 'momentum_5', 'momentum_10'
        ]

        flow_scaler_path = os.path.join(self.scaler_dir, "flow_scaler.bin")
        return_scaler_path = os.path.join(self.scaler_dir, "return_scaler.bin")
        price_scaler_path = os.path.join(self.scaler_dir, "price_scaler.bin")

        # 금액 지표를 상대적 지표로 대체 (거래대금 대비 비율)
        flow_cols_relative = []
        for col in self.flow_cols:
            if f'{col}_ratio' in df.columns:
                flow_cols_relative.append(f'{col}_ratio')
            else:
                # ratio가 없는 경우 원본 사용 (예: 등락률 등)
                if col in df.columns:
                    flow_cols_relative.append(col)
        
        # 스케일링 전략: 비율 데이터는 이미 정규화되어 있지만, 분산 차이를 줄이기 위해 가벼운 정규화 적용
        # StandardScaler 대신 RobustScaler 사용 (이상치에 강함)
        
        if is_train:
            # 학습 모드: fit_transform 후 저장
            # 상대적 지표(ratio)는 RobustScaler로 가벼운 정규화 (이상치에 강함)
            # 유효한 값이 있는 컬럼만 필터링 (모두 NaN인 컬럼 제외)
            flow_cols_valid = [col for col in flow_cols_relative if col in df.columns and df[col].notna().any()]
            if flow_cols_valid:
                df[flow_cols_valid] = self.flow_scaler.fit_transform(df[flow_cols_valid])
            
            # 등락률도 RobustScaler로 정규화 (유효한 값이 있는 경우만)
            if '등락률' in df.columns and df['등락률'].notna().any():
                df['등락률'] = self.return_scaler.fit_transform(df[['등락률']])
            
            # 상대적 가격 지표들도 RobustScaler로 정규화
            # 유효한 값이 있는 컬럼만 필터링
            available_price_cols = [col for col in price_relative_cols if col in df.columns and df[col].notna().any()]
            if available_price_cols:
                df[available_price_cols] = self.price_scaler.fit_transform(df[available_price_cols])
            
            joblib.dump(self.flow_scaler, flow_scaler_path)
            joblib.dump(self.return_scaler, return_scaler_path)
            joblib.dump(self.price_scaler, price_scaler_path)
            print(f"💾 스케일러 저장 완료: {self.scaler_dir}")
            print(f"   - 비율 데이터는 RobustScaler로 가벼운 정규화 적용 (분산 차이 조정)")
        else:
            # 추론 모드: load 후 transform만
            self.flow_scaler = joblib.load(flow_scaler_path)
            self.return_scaler = joblib.load(return_scaler_path)
            self.price_scaler = joblib.load(price_scaler_path)
            
            # 유효한 값이 있는 컬럼만 필터링
            flow_cols_valid = [col for col in flow_cols_relative if col in df.columns and df[col].notna().any()]
            if flow_cols_valid:
                df[flow_cols_valid] = self.flow_scaler.transform(df[flow_cols_valid])
            
            if '등락률' in df.columns and df['등락률'].notna().any():
                df['등락률'] = self.return_scaler.transform(df[['등락률']])
            
            available_price_cols = [col for col in price_relative_cols if col in df.columns and df[col].notna().any()]
            if available_price_cols:
                df[available_price_cols] = self.price_scaler.transform(df[available_price_cols])
            print("🔌 기존 스케일러 로드 및 적용 완료")

        return df

    # (이하 _add_technical_indicators, _add_date_features, _apply_clipping, _prepare_final_dataset는 기존과 동일)
    def _add_technical_indicators(self, df):
        # 1. 각 가격의 일일 수익률 (로그 수익률)
        for col in ['종가', '시가', '고가', '저가']:
            df[f'{col}_log_ret'] = np.log(df[col] / df[col].shift(1))
        
        # 2. 가격 차이 (상대적 지표)
        df['high_low_ratio'] = (df['고가'] - df['저가']) / df['종가']  # 당일 변동폭 비율
        df['open_close_ratio'] = (df['시가'] - df['종가']) / df['종가']  # 시가-종가 차이 비율
        df['high_close_ratio'] = (df['고가'] - df['종가']) / df['종가']  # 고가-종가 차이 비율
        df['low_close_ratio'] = (df['종가'] - df['저가']) / df['종가']  # 종가-저가 차이 비율
        
        # 3. 이동평균 (상대적 지표로 변환)
        df['MA5'] = df['종가'].rolling(window=5).mean()
        df['MA10'] = df['종가'].rolling(window=10).mean()
        df['MA20'] = df['종가'].rolling(window=20).mean()
        
        # 이동평균 대비 현재가 비율 (상대적)
        df['disparity_5'] = (df['종가'] / df['MA5']) - 1
        df['disparity_10'] = (df['종가'] / df['MA10']) - 1
        df['disparity_20'] = (df['종가'] / df['MA20']) - 1
        
        # 이동평균 간 차이 (상대적)
        df['ma_gap_5_10'] = (df['MA5'] / df['MA10']) - 1
        df['ma_gap_10_20'] = (df['MA10'] / df['MA20']) - 1
        
        # 이동평균 변화율 (상대적)
        df['ma5_gradient'] = np.log(df['MA5'] / df['MA5'].shift(1))
        df['ma10_gradient'] = np.log(df['MA10'] / df['MA10'].shift(1))
        
        # 4. 가격 모멘텀 (과거 N일 대비 변화율)
        df['momentum_3'] = (df['종가'] / df['종가'].shift(3)) - 1
        df['momentum_5'] = (df['종가'] / df['종가'].shift(5)) - 1
        df['momentum_10'] = (df['종가'] / df['종가'].shift(10)) - 1
        
        # 5. 금액 지표를 거래대금 대비로 정규화 (종목 간 일반화)
        # 거래대금이 0이거나 너무 작은 경우 방지
        df['거래대금_safe'] = df['거래대금'].replace(0, np.nan)
        
        # 각 금액 지표를 거래대금 대비 비율로 변환
        for col in self.flow_cols:
            if '금액' in col or '수량' in col:
                # 거래대금 대비 비율 (또는 거래량 대비 비율)
                if '금액' in col:
                    df[f'{col}_ratio'] = df[col] / df['거래대금_safe']
                else:  # 수량
                    df[f'{col}_ratio'] = df[col] / df['거래량'].replace(0, np.nan)
        
        # 매수/매도 관련 ratio에 Winsorization 적용 (1%, 99%)
        buy_sell_cols = [col for col in df.columns if '_ratio' in col and ('매수' in col or '매도' in col or '순매수' in col or '순매도' in col)]
        for col in buy_sell_cols:
            if col in df.columns and df[col].notna().sum() > 0:
                q01 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=q01, upper=q99)
        
        # 필수 컬럼만 체크하여 dropna (shift로 인한 NaN은 허용)
        # 필수: 종가, 거래량, 거래대금 등 핵심 데이터
        essential_cols = ['종가', '거래량', '거래대금']
        essential_cols = [col for col in essential_cols if col in df.columns]
        
        if essential_cols:
            print(f"  기술적 지표 추가 전: {len(df)}행")
            df = df.dropna(subset=essential_cols).reset_index(drop=True)
            print(f"  기술적 지표 추가 후: {len(df)}행 (필수 컬럼만 체크)")
            
            if len(df) == 0:
                print("  ⚠️ 경고: 필수 컬럼에 NaN이 많아 모든 행이 제거되었습니다.")
        else:
            df = df.reset_index(drop=True)
        
        return df

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
        # 매수/매도 관련 순매수 컬럼에 Winsorization 적용 (이미 ratio는 _add_technical_indicators에서 처리)
        # 순매수 관련 원본 컬럼이 남아있다면 여기서 처리
        for col in self.flow_cols:
            if col in df.columns and ('순매수' in col or '순매도' in col) and df[col].notna().any():
                q01 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(lower=q01, upper=q99)
        return df

    def _prepare_final_dataset(self, df):
        # 빈 데이터프레임 체크
        if len(df) == 0:
            print("  ⚠️ 경고: 데이터프레임이 비어있습니다.")
            return df
        
        # Target: 다음날 종가가 당일 종가보다 오르면 1, 아니면 0
        # 원본 종가를 사용 (아직 drop하지 않음)
        df['target'] = (df['종가'].shift(-1) > df['종가']).astype(int)
        df.loc[df.index[-1], 'target'] = np.nan
        
        # 원본 종가를 별도 컬럼으로 보존 (그래프용)
        df['original_close'] = df['종가'].copy()
        
        # target이 NaN인 행만 제거 (마지막 행)
        print(f"  최종 데이터셋 준비 전: {len(df)}행")
        df = df.dropna(subset=['target']).reset_index(drop=True)
        print(f"  최종 데이터셋 준비 후: {len(df)}행 (target만 체크)")
        
        if len(df) == 0:
            print("  ⚠️ 경고: target이 모두 NaN이어서 모든 행이 제거되었습니다.")
            return df
        
        # 절대 가격 제거 (우상향 편향 제거)
        # 상대적 지표만 유지 (original_close는 그래프용으로 유지)
        price_cols_to_drop = ['종가', '시가', '고가', '저가', 'MA5', 'MA10', 'MA20', '거래대금_safe']
        cols_to_drop = [col for col in price_cols_to_drop if col in df.columns]
        
        # 원본 절대 금액 컬럼도 제거 (ratio만 유지)
        for col in self.flow_cols:
            if col in df.columns and f'{col}_ratio' in df.columns:
                cols_to_drop.append(col)
        
        return df.drop(columns=cols_to_drop)