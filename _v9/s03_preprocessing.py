import pandas as pd
import numpy as np
import os

class StockPreprocessor:
    def __init__(self, stock_code=None, target_epsilon=0.005):
        self.stock_code = stock_code or "DEFAULT"
        self.target_epsilon = target_epsilon 
        self.ma_windows = [5, 20, 60]

    def run_pipeline(self, input_path, output_path, is_train=False):
        df = pd.read_csv(input_path)
        
        # 1. 날짜 정렬 및 날짜 피처 생성
        df['날짜'] = pd.to_datetime(df['날짜'].astype(str), errors='coerce')
        df = df.sort_values('날짜').reset_index(drop=True)
        df = self._add_date_features(df)
        
        # 2. 상대적 가격 및 변동성 피처
        df = self._calculate_relative_features(df)
        
        # 3. 체결 강도 (재현님 방식 유지)
        df = self._calculate_execution_strength(df)
        
        # 4. 거래량 및 수급 누적 피처 (정규화 포함)
        df = self._calculate_volume_investor_features(df)
        
        # 5. 기술적 지표 생성 (정규화 포함)
        df = self._calculate_technical_indicators(df)
        
        # 6. 타겟 레이블 생성 (Next Day Return 기반)
        df = self._create_target_labels(df)
        
        # 7. 데이터 클리닝 및 누수 방지 (당일 등락률 등 제거)
        df = self._finalize_dataset(df)
        df = df.dropna()
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 전처리 완료: {output_path} ({len(df)}행)")
        return df

    def _add_date_features(self, df):
        df['year_norm'] = (df['날짜'].dt.year - df['날짜'].dt.year.min()) / (df['날짜'].dt.year.max() - df['날짜'].dt.year.min() + 1e-8)
        df['year_sin'] = np.sin(2 * np.pi * df['year_norm'])
        df['year_cos'] = np.cos(2 * np.pi * df['year_norm'])
        df['month_sin'] = np.sin(2 * np.pi * df['날짜'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['날짜'].dt.month / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['날짜'].dt.day / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['날짜'].dt.day / 31)
        return df

    def _calculate_relative_features(self, df):
        # 시가 갭 및 당일 변동성
        df['open_gap'] = (df['시가'] / df['종가'].shift(1)) - 1
        df['high_ratio'] = (df['고가'] / df['시가']) - 1
        df['low_ratio'] = (df['저가'] / df['시가']) - 1
        df['volatility'] = (df['고가'] - df['저가']) / df['종가'] # 변동성 추가
        
        # 이동평균 이격도
        for w in self.ma_windows:
            ma = df['종가'].rolling(window=w).mean()
            df[f'gap_ma{w}'] = (df['종가'] / ma) - 1
        return df

    def _calculate_execution_strength(self, df):
        for inv in ['개인', '외국인', '기관계']:
            buy, sell = f'{inv}_매수수량', f'{inv}_매도수량'
            if buy in df.columns:
                total = df[buy] + df[sell]
                df[f'{inv}_체결강도'] = np.where(total > 0, df[buy] / total, 0.5)
        return df

    def _calculate_volume_investor_features(self, df):
        # 거래량 비율
        df['vol_ratio'] = df['거래량'].pct_change()
        # 무한대 값 클리핑 (이전 거래량이 0인 경우 방지)
        df['vol_ratio'] = df['vol_ratio'].replace([np.inf, -np.inf], np.nan)
        df['vol_ratio'] = np.clip(df['vol_ratio'], -1.0, 5.0)
        
        df['vol_ma5_ratio'] = (df['거래량'] / df['거래량'].rolling(5).mean()) - 1
        
        # 수급 누적 (거래량 대비 비중으로 정규화하여 Scale 통일)
        for inv in ['외국인', '기관계']:
            net_buy = f'{inv}_순매수수량'
            if net_buy in df.columns:
                # 5일 누적 순매수 비중
                df[f'{inv}_net_5d'] = df[net_buy].rolling(5).sum() / df['거래량'].rolling(5).sum()
                # 20일 누적 순매수 비중
                df[f'{inv}_net_20d'] = df[net_buy].rolling(20).sum() / df['거래량'].rolling(20).sum()
        return df

    def _calculate_technical_indicators(self, df):
        # RSI
        delta = df['종가'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain/(loss + 1e-8)))
        
        # MACD (가격 정규화: 종가로 나누어 비율화)
        ema12 = df['종가'].ewm(span=12).mean()
        ema26 = df['종가'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        df['macd_ratio'] = macd / df['종가']
        df['macd_diff_ratio'] = (macd - signal) / df['종가']
        
        # 볼린저 밴드 이격도
        ma20 = df['종가'].rolling(20).mean()
        std20 = df['종가'].rolling(20).std()
        df['bb_upper_ratio'] = (df['종가'] / (ma20 + 2*std20)) - 1
        df['bb_lower_ratio'] = (df['종가'] / (ma20 - 2*std20)) - 1
        return df

    def _create_target_labels(self, df):
        # 오늘 종가 → 내일 종가 수익률
        # next_rtn = (내일 종가 - 오늘 종가) / 오늘 종가
        df['next_rtn'] = df['종가'].shift(-1) / df['종가'] - 1
        
        # 마지막 행은 next_rtn이 NaN이므로 target도 NaN으로 설정
        df['target'] = (df['next_rtn'] > 0).astype(float)  # float로 변경하여 NaN 허용
        return df

    def _finalize_dataset(self, df):
        # 데이터 누수 방지: 당일 등락률, 전일대비 등 제거
        inv_cols = [c for c in df.columns if '매수' in c or '매도' in c]

        # print(df.columns)

        drop_cols = [
            '종가', '시가', '고가', '저가', '거래량', '거래대금', '대비부호', '전일대비', '등락률',
            # 1. 원본 데이터 및 중복 데이터 (이미 피처화됨)
            '락구분', '분할비율', '수정주가여부', '재평가사유', 'bold_yn', 'year_norm',
            
            # 2. _investor 접미사가 붙은 중복 시세 데이터 (데이터 누수 위험 및 중복)
            'stck_clpr_investor', 'prdy_vrss_investor', 'prdy_vrss_sign_investor', 
            'acml_vol_investor', 'acml_tr_pbmn_investor', 'stck_oprc_investor', 
            'stck_hgpr_investor', 'stck_lwpr_investor',
             '외국인_net_5d', '외국인_net_20d', '기관계_net_5d', '기관계_net_20d',
            
            # 3. 세부 투자자별 원본 수치 (이미 정규화된 net_5d/20d 및 체결강도로 변환됨)
            'frgn_reg_askp_qty', 'frgn_reg_bidp_qty', 'frgn_reg_askp_pbmn', 'frgn_reg_bidp_pbmn',
            'frgn_nreg_askp_qty', 'frgn_nreg_bidp_qty', 'frgn_nreg_askp_pbmn', 'frgn_nreg_bidp_pbmn',
            'scrt_seln_tr_pbmn', 'scrt_shnu_tr_pbmn', 'ivtr_seln_tr_pbmn', 'ivtr_shnu_tr_pbmn',
            'pe_fund_seln_tr_pbmn', 'pe_fund_shnu_tr_pbmn', 'bank_seln_tr_pbmn', 'bank_shnu_tr_pbmn',
            'insu_seln_tr_pbmn', 'insu_shnu_tr_pbmn', 'mrbn_seln_tr_pbmn', 'mrbn_shnu_tr_pbmn',
            'fund_seln_tr_pbmn', 'fund_shnu_tr_pbmn', 'etc_seln_tr_pbmn', 'etc_shnu_tr_pbmn',
            'etc_orgt_seln_tr_pbmn', 'etc_orgt_shnu_tr_pbmn', 'etc_corp_seln_tr_pbmn', 'etc_corp_shnu_tr_pbmn'
        ]
        
        df = df.drop(columns=[c for c in drop_cols + inv_cols if c in df.columns])
        
        # target이 NaN인 행만 제거 (마지막 행은 예측 불가능하므로 제거)
        # 다른 컬럼의 NaN은 허용하되, target이 있는 행만 유지
        if 'target' in df.columns:
            df = df.dropna(subset=['target']).reset_index(drop=True)
            # target을 다시 int로 변환
            df['target'] = df['target'].astype(int)
        else:
            # target이 없으면 전체 dropna (이 경우는 없어야 함)
            df = df.dropna().reset_index(drop=True)
        
        return df