"""
XGBoost + LightGBM + CatBoost 앙상블 학습
TimeSeriesSplit 기반 검증
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import os
from datetime import datetime


class StockEnsembleTrainer:
    def __init__(self, data_path, n_splits=5, lag_days=[1, 2, 3, 5, 10], 
                 target_threshold=0.01):
        """
        Args:
            data_path: 데이터 경로
            n_splits: K-Fold 수
            lag_days: Lag 피처 생성할 일수
            target_threshold: 타겟 임계값 (0.01 = 1%)
                - None이면 기존 타겟 사용 (0보다 크면 1)
                - 0.01이면 1% 이상 상승=1, 1% 이상 하락=0, 나머지 제외
        """
        self.data_path = data_path
        self.n_splits = n_splits
        self.lag_days = lag_days
        self.target_threshold = target_threshold
        self.models = {}
        self.label_encoder = LabelEncoder()
        
        # 결과 저장 디렉토리
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
        
    def load_data(self):
        """데이터 로드 및 전처리"""
        print("📂 데이터 로드 중...")
        df = pd.read_csv(self.data_path)
        
        # 날짜 정렬
        df['날짜'] = pd.to_datetime(df['날짜'])
        df = df.sort_values(['stock_code', '날짜']).reset_index(drop=True)
        
        print(f"   원본 데이터: {len(df):,}건")
        print(f"   기간: {df['날짜'].min().date()} ~ {df['날짜'].max().date()}")
        print(f"   종목 수: {df['stock_code'].nunique()}개")
        
        # 타겟 재정의
        if self.target_threshold is not None:
            df = self.redefine_target(df)
        
        # Lag 피처 추가
        df = self.add_lag_features(df)
        
        print(f"   Lag 피처 추가 후: {len(df):,}건")
        print(f"   클래스 분포: 0={len(df[df['target']==0]):,}, 1={len(df[df['target']==1]):,}")
        
        return df
    
    def redefine_target(self, df):
        """타겟 재정의: 임계값 이상 상승/하락만 사용
        
        next_rtn = (내일 종가 - 오늘 종가) / 오늘 종가
        → 오늘 종가 기준 내일 종가 수익률
        """
        print(f"\n🎯 타겟 재정의 (종가→다음날 종가, 임계값: ±{self.target_threshold*100:.1f}%)")
        
        # next_rtn이 이미 전처리 데이터에 포함되어 있음
        if 'next_rtn' not in df.columns:
            raise ValueError("next_rtn 컬럼이 없습니다. 전처리를 다시 실행하세요.")
        
        # NaN 제거 (마지막 행)
        df = df.dropna(subset=['next_rtn'])
        
        before_filter = len(df)
        
        # 타겟 재정의
        # 임계값 이상 상승 = 1 (시가 대비 종가 +1% 이상)
        # 임계값 이상 하락 = 0 (시가 대비 종가 -1% 이하)
        # 그 사이 = 제외 (노이즈)
        df_up = df[df['next_rtn'] >= self.target_threshold].copy()
        df_up['target'] = 1
        
        df_down = df[df['next_rtn'] <= -self.target_threshold].copy()
        df_down['target'] = 0
        
        df_filtered = pd.concat([df_up, df_down], ignore_index=True)
        df_filtered = df_filtered.sort_values(['stock_code', '날짜']).reset_index(drop=True)
        
        # next_rtn 컬럼 제거 (피처로 사용하면 데이터 누수!)
        df_filtered = df_filtered.drop(columns=['next_rtn'])
        
        after_filter = len(df_filtered)
        removed = before_filter - after_filter
        
        print(f"   상승 (종가→종가 ≥+{self.target_threshold*100:.1f}%): {len(df_up):,}건")
        print(f"   하락 (종가→종가 ≤-{self.target_threshold*100:.1f}%): {len(df_down):,}건")
        print(f"   제외 (노이즈 구간): {removed:,}건 ({removed/before_filter*100:.1f}%)")
        print(f"   필터링 후: {after_filter:,}건")
        
        return df_filtered
    
    def add_lag_features(self, df):
        """Lag 피처 추가 (종목별로)"""
        print(f"📊 Lag 피처 생성 중... (lag_days: {self.lag_days})")
        
        # Lag 피처를 만들 컬럼들 (중요한 피처들)
        lag_cols = [
            'open_gap', 'high_ratio', 'low_ratio', 'volatility',
            'gap_ma5', 'gap_ma20', 'gap_ma60',
            '개인_체결강도', '외국인_체결강도', '기관계_체결강도',
            'vol_ratio', 'vol_ma5_ratio', 'rsi',
            'macd_ratio', 'macd_diff_ratio',
            'bb_upper_ratio', 'bb_lower_ratio'
        ]
        
        # 종목별로 Lag 피처 생성
        lag_dfs = []
        
        for stock_code in df['stock_code'].unique():
            stock_df = df[df['stock_code'] == stock_code].copy()
            stock_df = stock_df.sort_values('날짜')
            
            # 각 Lag에 대해 피처 생성
            for lag in self.lag_days:
                for col in lag_cols:
                    if col in stock_df.columns:
                        stock_df[f'{col}_lag{lag}'] = stock_df[col].shift(lag)
            
            # 변화율 피처 추가 (1일 전 대비)
            for col in ['gap_ma5', 'rsi', '외국인_체결강도', '기관계_체결강도']:
                if col in stock_df.columns:
                    stock_df[f'{col}_change'] = stock_df[col] - stock_df[col].shift(1)
            
            lag_dfs.append(stock_df)
        
        df_with_lag = pd.concat(lag_dfs, ignore_index=True)
        
        # NaN 제거 (Lag로 인해 앞부분에 NaN 생김)
        before_drop = len(df_with_lag)
        df_with_lag = df_with_lag.dropna().reset_index(drop=True)
        after_drop = len(df_with_lag)
        
        print(f"   생성된 Lag 피처 수: {len([c for c in df_with_lag.columns if 'lag' in c or 'change' in c])}개")
        print(f"   NaN 제거: {before_drop - after_drop:,}건 제거됨")
        
        return df_with_lag
    
    def prepare_features(self, df):
        """피처 준비"""
        # 제외할 컬럼
        exclude_cols = ['날짜', 'target', 'stock_code', 'stock_name']
        
        # 피처 컬럼
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['target'].values
        dates = df['날짜'].values
        stock_codes = df['stock_code'].values
        
        # 종목 코드를 숫자로 인코딩 (피처로 사용)
        stock_code_encoded = self.label_encoder.fit_transform(stock_codes)
        X['stock_code_encoded'] = stock_code_encoded
        
        print(f"\n📊 피처 정보:")
        print(f"   피처 수: {len(X.columns)}개")
        print(f"   피처 목록: {list(X.columns)}")
        
        return X, y, dates, stock_codes
    
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """XGBoost 학습"""
        print("\n🔵 XGBoost 학습 중...")
        
        # 클래스 가중치 계산
        scale_pos = len(y_train[y_train==0]) / len(y_train[y_train==1])
        
        model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos,
            random_state=42,
            n_jobs=-1,
            eval_metric='auc',
            early_stopping_rounds=100
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100
        )
        
        print(f"   Best iteration: {model.best_iteration}")
        self.models['xgboost'] = model
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """LightGBM 학습"""
        print("\n🟢 LightGBM 학습 중...")
        
        # 클래스 가중치 계산
        scale_pos = len(y_train[y_train==0]) / len(y_train[y_train==1])
        
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=30,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos,
            random_state=42,
            n_jobs=-1,
            verbose=100
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100, verbose=True)]
        )
        
        print(f"   Best iteration: {model.best_iteration_}")
        self.models['lightgbm'] = model
        return model
    
    def train_catboost(self, X_train, y_train, X_val, y_val):
        """CatBoost 학습"""
        print("\n🟡 CatBoost 학습 중...")
        
        # 클래스 가중치 계산
        scale_pos = len(y_train[y_train==0]) / len(y_train[y_train==1])
        
        model = CatBoostClassifier(
            iterations=1000,
            depth=8,
            learning_rate=0.01,
            l2_leaf_reg=3,
            scale_pos_weight=scale_pos,
            random_seed=42,
            verbose=100,
            early_stopping_rounds=100,
            eval_metric='AUC'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=100
        )
        
        print(f"   Best iteration: {model.best_iteration_}")
        self.models['catboost'] = model
        return model
    
    def ensemble_predict(self, X, method='soft'):
        """앙상블 예측"""
        predictions = {}
        probas = {}
        
        for name, model in self.models.items():
            pred = model.predict(X)
            proba = model.predict_proba(X)[:, 1]
            predictions[name] = pred
            probas[name] = proba
        
        if method == 'soft':
            # Soft Voting: 확률 평균
            avg_proba = np.mean([probas[name] for name in self.models], axis=0)
            ensemble_pred = (avg_proba >= 0.5).astype(int)
            ensemble_proba = avg_proba
        else:
            # Hard Voting: 다수결
            all_preds = np.array([predictions[name] for name in self.models])
            ensemble_pred = (np.mean(all_preds, axis=0) >= 0.5).astype(int)
            ensemble_proba = np.mean([probas[name] for name in self.models], axis=0)
        
        return ensemble_pred, ensemble_proba, predictions, probas
    
    def evaluate(self, y_true, y_pred, y_proba, model_name="Model"):
        """성능 평가"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_proba)
        }
        
        print(f"\n📈 {model_name} 성능:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1 Score:  {metrics['f1']:.4f}")
        print(f"   AUC:       {metrics['auc']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, feature_names):
        """피처 중요도 추출"""
        importance_df = pd.DataFrame({'feature': feature_names})
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df[f'{name}_importance'] = model.feature_importances_
        
        # 평균 중요도 계산
        imp_cols = [c for c in importance_df.columns if 'importance' in c]
        importance_df['avg_importance'] = importance_df[imp_cols].mean(axis=1)
        importance_df = importance_df.sort_values('avg_importance', ascending=False)
        
        return importance_df
    
    def save_models(self, suffix=''):
        """모델 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, model in self.models.items():
            path = f'{self.model_dir}/{name}_{timestamp}{suffix}.pkl'
            joblib.dump(model, path)
            print(f"💾 {name} 저장: {path}")
        
        # Label Encoder 저장
        le_path = f'{self.model_dir}/label_encoder_{timestamp}{suffix}.pkl'
        joblib.dump(self.label_encoder, le_path)
        print(f"💾 LabelEncoder 저장: {le_path}")
    
    def run(self):
        """TimeSeriesSplit 교차 검증 학습"""
        print("=" * 60)
        print(f"🚀 앙상블 모델 학습 (TimeSeries {self.n_splits}-Fold)")
        print("=" * 60)
        
        # 1. 데이터 로드
        df = self.load_data()
        
        # 2. 피처 준비
        X, y, dates, stock_codes = self.prepare_features(df)
        
        # 날짜 기준 정렬 (TimeSeriesSplit을 위해)
        df_sorted = df.sort_values('날짜').reset_index(drop=True)
        X = X.loc[df_sorted.index].reset_index(drop=True)
        y = y[df_sorted.index]
        dates = dates[df_sorted.index]
        
        X_np = X.values  # numpy로 변환
        
        # K-Fold 결과 저장
        fold_results = {
            'xgboost': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []},
            'catboost': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []},
            'ensemble': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
        }
        
        # 3. TimeSeriesSplit 교차 검증
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_np), 1):
            print(f"\n{'='*60}")
            print(f"📂 Fold {fold}/{self.n_splits}")
            print(f"{'='*60}")
            
            X_train, X_val = X_np[train_idx], X_np[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 날짜 범위 출력
            train_dates = dates[train_idx]
            val_dates = dates[val_idx]
            print(f"   학습: {pd.Timestamp(train_dates[0]).date()} ~ {pd.Timestamp(train_dates[-1]).date()} ({len(X_train):,}건)")
            print(f"   검증: {pd.Timestamp(val_dates[0]).date()} ~ {pd.Timestamp(val_dates[-1]).date()} ({len(X_val):,}건)")
            print(f"   학습 클래스 비율 - 0: {sum(y_train==0):,}, 1: {sum(y_train==1):,}")
            
            # 모델 초기화
            self.models = {}
            
            # 모델 학습
            self.train_xgboost(X_train, y_train, X_val, y_val)
            self.train_catboost(X_train, y_train, X_val, y_val)
            
            # 개별 모델 평가
            print(f"\n📊 Fold {fold} 성능:")
            for name, model in self.models.items():
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]
                metrics = self.evaluate(y_val, y_pred, y_proba, name.upper())
                for k, v in metrics.items():
                    fold_results[name][k].append(v)
            
            # 앙상블 평가
            ensemble_pred, ensemble_proba, _, _ = self.ensemble_predict(X_val, method='soft')
            ensemble_metrics = self.evaluate(y_val, ensemble_pred, ensemble_proba, "ENSEMBLE")
            for k, v in ensemble_metrics.items():
                fold_results['ensemble'][k].append(v)
        
        # 4. 전체 데이터로 최종 모델 학습
        print(f"\n{'='*60}")
        print(f"🎯 최종 모델 학습 (전체 데이터)")
        print(f"{'='*60}")
        
        self.models = {}
        # 전체 데이터의 80%로 학습, 20%로 검증 (최종 모델용)
        split_idx = int(len(X_np) * 0.8)
        X_train_final, X_val_final = X_np[:split_idx], X_np[split_idx:]
        y_train_final, y_val_final = y[:split_idx], y[split_idx:]
        
        self.train_xgboost(X_train_final, y_train_final, X_val_final, y_val_final)
        self.train_catboost(X_train_final, y_train_final, X_val_final, y_val_final)
        
        # 5. K-Fold 평균 결과
        print(f"\n{'='*60}")
        print(f"📋 {self.n_splits}-Fold 교차 검증 결과 (평균 ± 표준편차)")
        print(f"{'='*60}")
        
        avg_results = {}
        for model_name, metrics in fold_results.items():
            avg_results[model_name] = {}
            print(f"\n📈 {model_name.upper()}:")
            for metric_name, values in metrics.items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                avg_results[model_name][metric_name] = mean_val
                print(f"   {metric_name:10s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # 6. 피처 중요도
        print(f"\n{'='*60}")
        print(f"📌 피처 중요도 (Top 10)")
        print(f"{'='*60}")
        
        importance_df = self.get_feature_importance(X.columns.tolist())
        print(importance_df[['feature', 'avg_importance']].head(10).to_string(index=False))
        
        # 7. 모델 저장
        print(f"\n{'='*60}")
        print(f"💾 모델 저장")
        print(f"{'='*60}")
        
        self.save_models()
        importance_df.to_csv(f'{self.model_dir}/feature_importance.csv', index=False)
        
        print("\n✅ 학습 완료!")
        
        return avg_results, importance_df


def main():
    # 데이터 경로
    data_path = '_data/merged_all_stocks_20260131.csv'
    
    # 학습 실행
    # target_threshold: 1% 이상 상승/하락만 학습
    # None으로 설정하면 기존 타겟 사용
    trainer = StockEnsembleTrainer(
        data_path, 
        n_splits=5,
        lag_days=[1, 2, 3, 5, 10],
        target_threshold=0.01  # 1% 임계값
    )
    metrics, importance = trainer.run()
    
    print("\n✅ 학습 완료!")


if __name__ == "__main__":
    main()

