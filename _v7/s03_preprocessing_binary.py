import pandas as pd
import numpy as np
import os

class StockPreprocessorBinary:
    def __init__(self, stock_code=None, target_epsilon=0.005):
        """
        이진 분류용 전처리기 (상승/하락만 예측, 보합 제거)
        
        Args:
            target_epsilon: 보합(1)을 결정하는 수익률 임계값 (예: 0.005는 +-0.5%)
                           이 범위 내의 데이터는 제거됨
        """
        self.stock_code = stock_code or "DEFAULT"
        self.target_epsilon = target_epsilon
        self.ma_windows = [5, 20, 60] # 이동평균 윈도우

    def run_pipeline(self, input_path, output_path, is_train=False):
        df = pd.read_csv(input_path)
        print(f"📥 원본 데이터: {len(df)}행")
        
        # 1. 날짜 정렬 및 날짜 피처 생성
        df['날짜'] = pd.to_datetime(df['날짜'].astype(str), errors='coerce')
        df = df.sort_values('날짜').reset_index(drop=True)
        print(f"📅 날짜 정렬 후: {len(df)}행")
        
        # 날짜에서 연/월/일 추출 및 sin/cos 변환
        df = self._add_date_features(df)
        
        # 2. 상대적 가격 피처 생성 (OHLC 정규화 문제 해결)
        df = self._calculate_relative_features(df)
        print(f"📊 피처 생성 후: {len(df)}행")
        
        # 3. 체결 강도 계산 (재현님의 비중 방식)
        df = self._calculate_execution_strength(df)
        print(f"💪 체결강도 계산 후: {len(df)}행")
        
        # 4. 타겟 레이블 생성 (이진 분류: 0=하락, 1=상승, 보합 제거)
        df = self._create_target_labels(df)
        print(f"🎯 타겟 생성 후: {len(df)}행")
        
        # 5. 최종 데이터 정리 (불필요한 원본 OHLC 제거 및 결측치 처리)
        # 맨 앞의 NaN 값 제거 (shift 연산으로 인한 첫 행의 NaN 등)
        before_drop = len(df)
        
        # 필수 컬럼 결정: target은 항상 필수
        required_cols = ['target']
        
        # target이 NaN인 행 제거 (마지막 행 등)
        df = df.dropna(subset=required_cols)
        
        # 앞쪽에 남아있는 NaN 값들 제거 (open_gap 등 shift로 인한 NaN)
        # open_gap이 NaN인 행들을 제거 (첫 행의 shift로 인한 NaN)
        if 'open_gap' in df.columns:
            before_open_drop = len(df)
            df = df.dropna(subset=['open_gap'])
            after_open_drop = len(df)
            if before_open_drop > after_open_drop:
                print(f"🔝 앞쪽 NaN 행 제거: {before_open_drop}행 → {after_open_drop}행 (open_gap NaN 제거)")
        
        after_drop = len(df)
        print(f"🧹 결측치 제거: {before_drop}행 → {after_drop}행 (필수 컬럼: {required_cols})")
        
        if len(df) == 0:
            print("⚠️ 경고: 모든 행이 제거되었습니다. 데이터 기간이 너무 짧거나 필수 컬럼에 결측치가 많습니다.")
            print(f"   원본 데이터 행 수: {before_drop}")
            print(f"   필수 컬럼: {required_cols}")
        
        # 6. ma5, ma20, ma60 컬럼 제거 (gap_ma는 유지)
        ma_cols_to_drop = ['ma5', 'ma20', 'ma60']
        for col in ma_cols_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
        print(f"🗑️ 이동평균 컬럼 제거: {ma_cols_to_drop}")
        
        # 7. 사용하지 않는 컬럼 제거 (필요한 피처만 유지)
        feature_cols = [
            '날짜',  # 참고용
            # 날짜 sin/cos 피처
            'year_sin', 'year_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
            # 상대적 가격 피처
            'open_gap', 'high_ratio', 'low_ratio',
            # 이동평균 이격도 피처 (gap_ma만 유지)
            'gap_ma5', 'gap_ma20', 'gap_ma60',
            # 체결강도 피처
            '개인_체결강도', '외국인_체결강도', '기관계_체결강도',
            # 타겟
            'next_rtn', 'target'
        ]
        
        # 존재하는 컬럼만 선택
        available_cols = [col for col in feature_cols if col in df.columns]
        df = df[available_cols]
        df = df.dropna()
        
        print(f"🗑️ 불필요한 컬럼 제거 완료: {len(available_cols)}개 피처만 유지")
        
        # 타겟 분포 확인
        if 'target' in df.columns:
            target_counts = df['target'].value_counts().sort_index()
            print(f"📊 최종 타겟 분포: {dict(target_counts)} (0:하락, 1:상승)")
        
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 전처리 완료: {output_path} ({len(df)}행 × {len(df.columns)}컬럼)")
        return df

    def _add_date_features(self, df):
        """날짜에서 연/월/일을 추출하고 sin/cos 변환 피처 생성"""
        print("📅 날짜 피처 생성 중 (연/월/일 sin/cos 변환)...")
        
        # 연/월/일 추출
        df['year'] = df['날짜'].dt.year
        df['month'] = df['날짜'].dt.month
        df['day'] = df['날짜'].dt.day
        
        # 연도 sin/cos 변환 (연도를 0-1 범위로 정규화)
        # 연도 범위를 최소/최대로 정규화
        year_min = df['year'].min()
        year_max = df['year'].max()
        year_normalized = (df['year'] - year_min) / (year_max - year_min + 1e-8)
        df['year_sin'] = np.sin(2 * np.pi * year_normalized)
        df['year_cos'] = np.cos(2 * np.pi * year_normalized)
        
        # 월 sin/cos 변환 (1-12 → 0-2π)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 일 sin/cos 변환 (1-31 → 0-2π, 각 월의 최대 일수 고려)
        # 간단하게 31일 기준으로 변환 (각 월의 실제 일수는 다르지만 일반적으로 사용)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        
        # 임시 컬럼 제거
        df = df.drop(columns=['year', 'month', 'day'])
        
        return df

    def _calculate_relative_features(self, df):
        """OHLC 가격을 비율 및 이격도로 변환"""
        print("📊 상대적 가격 피처 생성 중...")
        
        # 시가 갭 (전일 종가 대비)
        df['open_gap'] = (df['시가'] / df['종가'].shift(1)) - 1
        
        # 당일 고가/저가 비율 (당일 시가 대비)
        df['high_ratio'] = (df['고가'] / df['시가']) - 1
        df['low_ratio'] = (df['저가'] / df['시가']) - 1
        
        # 이동평균 이격도 (MA Gap) 계산 - gap_ma를 위해 ma 계산 후 gap만 사용
        data_len = len(df)
        for w in self.ma_windows:
            if w <= data_len:
                ma_col = f'ma{w}'
                df[ma_col] = df['종가'].rolling(window=w).mean()
                df[f'gap_ma{w}'] = (df['종가'] / df[ma_col]) - 1
            else:
                print(f"⚠️ 데이터 길이({data_len})가 이동평균 윈도우({w})보다 짧아 {w}일 이동평균을 건너뜁니다.")
                # 짧은 데이터의 경우 gap_ma만 NaN으로 설정
                df[f'gap_ma{w}'] = np.nan
            
        return df

    def _calculate_execution_strength(self, df):
        """매수 비중 방식 체결강도 계산 (0~1 사이)"""
        investors = ['개인', '외국인', '기관계']
        for inv in investors:
            buy_col, sell_col = f'{inv}_매수수량', f'{inv}_매도수량'
            if buy_col in df.columns and sell_col in df.columns:
                total = df[buy_col] + df[sell_col]
                df[f'{inv}_체결강도'] = np.where(total > 0, df[buy_col] / total, 0.5)
        return df

    def _create_target_labels(self, df):
        """이진 분류 타겟 생성 (내일 종가 수익률 기준, 보합 제거)"""
        print(f"🎯 이진 분류 타겟 레이블 생성 중 (Epsilon: {self.target_epsilon}, 보합 제거)...")
        
        # 내일 수익률 계산
        df['next_rtn'] = df['종가'].pct_change().shift(-1)
        
        # 이진 분류 레이블링
        # 1: 상승 (> epsilon), 0: 하락 (< -epsilon), 보합은 제거
        conditions = [
            (df['next_rtn'] > self.target_epsilon),   # 상승
            (df['next_rtn'] < -self.target_epsilon)   # 하락
        ]
        choices = [1, 0]  # 1=상승, 0=하락
        df['target'] = np.select(conditions, choices, default=np.nan)  # 보합은 NaN으로 설정
        
        # 보합 데이터 제거
        before_remove = len(df)
        df = df.dropna(subset=['target'])
        after_remove = len(df)
        removed_count = before_remove - after_remove
        
        if removed_count > 0:
            print(f"   보합 데이터 제거: {removed_count}행 제거됨 ({before_remove}행 → {after_remove}행)")
        
        # 타겟 분포 확인
        if 'target' in df.columns:
            target_counts = df['target'].value_counts().sort_index()
            print(f"   타겟 분포: {dict(target_counts)} (0:하락, 1:상승)")
        
        # next_rtn 컬럼 제거 (타겟만 남김)
        df = df.drop(columns=['next_rtn'])
        
        return df









