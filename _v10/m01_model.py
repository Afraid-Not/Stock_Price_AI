"""
주가 예측 AI 모델 (앙상블: XGBoost + LightGBM + CatBoost)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib
import os
from datetime import datetime


class StockPredictionModel:
    """주가 상승/하락 예측 앙상블 모델"""
    
    def __init__(self, model_dir="D:/stock/_v10/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # 앙상블 모델들
        self.models = {
            'xgb': None,
            'lgb': None,
            'cat': None
        }
        
        # 앙상블 가중치 (학습 후 업데이트)
        self.weights = {'xgb': 0.35, 'lgb': 0.35, 'cat': 0.30}
        
        # 피처 중요도
        self.feature_importance = None
        
    def _get_xgb_model(self):
        """XGBoost 모델 설정"""
        return xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            early_stopping_rounds=50
        )
    
    def _get_lgb_model(self):
        """LightGBM 모델 설정"""
        return lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1
        )
    
    def _get_cat_model(self):
        """CatBoost 모델 설정"""
        return CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_seed=42,
            verbose=False,
            early_stopping_rounds=50
        )
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """모델 학습"""
        print("=" * 60)
        print("🚀 모델 학습 시작")
        print("=" * 60)
        
        # Validation 데이터가 없으면 train에서 분리
        if X_val is None:
            split_idx = int(len(X_train) * 0.85)
            X_val = X_train.iloc[split_idx:]
            y_val = y_train.iloc[split_idx:]
            X_train = X_train.iloc[:split_idx]
            y_train = y_train.iloc[:split_idx]
        
        # 1. XGBoost 학습
        print("\n📊 [1/3] XGBoost 학습 중...")
        self.models['xgb'] = self._get_xgb_model()
        self.models['xgb'].fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        xgb_pred = self.models['xgb'].predict_proba(X_val)[:, 1]
        xgb_auc = roc_auc_score(y_val, xgb_pred)
        print(f"   ✅ XGBoost AUC: {xgb_auc:.4f}")
        
        # 2. LightGBM 학습
        print("\n📊 [2/3] LightGBM 학습 중...")
        self.models['lgb'] = self._get_lgb_model()
        self.models['lgb'].fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        lgb_pred = self.models['lgb'].predict_proba(X_val)[:, 1]
        lgb_auc = roc_auc_score(y_val, lgb_pred)
        print(f"   ✅ LightGBM AUC: {lgb_auc:.4f}")
        
        # 3. CatBoost 학습
        print("\n📊 [3/3] CatBoost 학습 중...")
        self.models['cat'] = self._get_cat_model()
        self.models['cat'].fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )
        cat_pred = self.models['cat'].predict_proba(X_val)[:, 1]
        cat_auc = roc_auc_score(y_val, cat_pred)
        print(f"   ✅ CatBoost AUC: {cat_auc:.4f}")
        
        # 가중치 업데이트 (성능 기반)
        total_auc = xgb_auc + lgb_auc + cat_auc
        self.weights = {
            'xgb': xgb_auc / total_auc,
            'lgb': lgb_auc / total_auc,
            'cat': cat_auc / total_auc
        }
        
        # 앙상블 성능 평가
        ensemble_pred = self._ensemble_predict_proba(X_val)
        ensemble_auc = roc_auc_score(y_val, ensemble_pred)
        ensemble_acc = accuracy_score(y_val, (ensemble_pred > 0.5).astype(int))
        
        print("\n" + "=" * 60)
        print("📈 학습 결과 요약")
        print("=" * 60)
        print(f"   XGBoost AUC:  {xgb_auc:.4f} (가중치: {self.weights['xgb']:.2%})")
        print(f"   LightGBM AUC: {lgb_auc:.4f} (가중치: {self.weights['lgb']:.2%})")
        print(f"   CatBoost AUC: {cat_auc:.4f} (가중치: {self.weights['cat']:.2%})")
        print(f"   ─────────────────────────────────")
        print(f"   🏆 앙상블 AUC: {ensemble_auc:.4f}")
        print(f"   🏆 앙상블 ACC: {ensemble_acc:.4f}")
        print("=" * 60)
        
        # 피처 중요도 계산
        self._calculate_feature_importance(X_train.columns)
        
        return {
            'xgb_auc': xgb_auc,
            'lgb_auc': lgb_auc,
            'cat_auc': cat_auc,
            'ensemble_auc': ensemble_auc,
            'ensemble_acc': ensemble_acc
        }
    
    def _ensemble_predict_proba(self, X):
        """앙상블 확률 예측"""
        preds = []
        for name, model in self.models.items():
            if model is not None:
                pred = model.predict_proba(X)[:, 1]
                preds.append(pred * self.weights[name])
        return np.sum(preds, axis=0)
    
    def predict(self, X):
        """예측 (0: 하락, 1: 상승)"""
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
    
    def predict_proba(self, X):
        """상승 확률 예측 (0~1)"""
        return self._ensemble_predict_proba(X)
    
    def _calculate_feature_importance(self, feature_names):
        """피처 중요도 계산 (앙상블 평균)"""
        importance_dict = {}
        
        for name, model in self.models.items():
            if model is not None:
                if name == 'xgb':
                    imp = model.feature_importances_
                elif name == 'lgb':
                    imp = model.feature_importances_
                elif name == 'cat':
                    imp = model.feature_importances_
                
                for i, feat in enumerate(feature_names):
                    if feat not in importance_dict:
                        importance_dict[feat] = 0
                    importance_dict[feat] += imp[i] * self.weights[name]
        
        self.feature_importance = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False)
        
        return self.feature_importance
    
    def save(self, suffix=""):
        """모델 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.model_dir, f"ensemble_model_{timestamp}{suffix}.pkl")
        
        save_data = {
            'models': self.models,
            'weights': self.weights,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(save_data, save_path)
        print(f"💾 모델 저장 완료: {save_path}")
        return save_path
    
    def load(self, model_path):
        """모델 로드"""
        save_data = joblib.load(model_path)
        self.models = save_data['models']
        self.weights = save_data['weights']
        self.feature_importance = save_data.get('feature_importance')
        print(f"📂 모델 로드 완료: {model_path}")


class ModelEvaluator:
    """모델 성능 평가 및 백테스트"""
    
    @staticmethod
    def evaluate(y_true, y_pred, y_proba=None):
        """분류 성능 평가"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        
        return metrics
    
    @staticmethod
    def backtest_simple(df, predictions, initial_capital=10_000_000):
        """
        간단한 백테스트
        df: 'next_rtn' 컬럼 필요 (다음 날 수익률)
        predictions: 예측값 (1: 매수, 0: 관망)
        """
        capital = initial_capital
        returns = []
        
        for i in range(len(predictions)):
            if predictions[i] == 1:  # 매수 시그널
                daily_return = df['next_rtn'].iloc[i]
                capital *= (1 + daily_return)
            returns.append(capital)
        
        final_return = (capital / initial_capital - 1) * 100
        
        # 벤치마크 (Buy & Hold)
        buy_hold_return = (np.prod(1 + df['next_rtn'].values) - 1) * 100
        
        return {
            'final_capital': capital,
            'total_return': final_return,
            'buy_hold_return': buy_hold_return,
            'alpha': final_return - buy_hold_return,
            'capital_history': returns
        }


if __name__ == "__main__":
    # 테스트 코드
    print("모델 모듈 로드 테스트 완료!")

