import pandas as pd
import numpy as np
import os
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class StockModelTrainer:
    def __init__(self, data_path, model_dir="models", test_size=0.2, val_size=0.2, 
                 n_splits=5, random_state=42):
        """
        Args:
            data_path: 전처리된 CSV 파일 경로
            model_dir: 모델 저장 디렉토리
            test_size: 테스트 데이터 비율
            val_size: Validation 데이터 비율 (Train 데이터 기준)
            n_splits: K-Fold 교차 검증 fold 수
            random_state: 랜덤 시드
        """
        self.data_path = data_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = test_size
        self.val_size = val_size
        self.n_splits = n_splits
        self.random_state = random_state
        
        # K-Fold 교차 검증용
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # 모델 저장용
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.cv_results = {}  # 교차 검증 결과
        
    def load_data(self):
        """데이터 로드 및 전처리 (Train/Val/Test 분할)"""
        print(f"📂 데이터 로드: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # 날짜 컬럼 제거 (피처로 사용하지 않음)
        if '날짜' in df.columns:
            df = df.drop(columns=['날짜'])
        
        # Target과 Feature 분리
        if 'target' not in df.columns:
            raise ValueError("'target' 컬럼이 없습니다.")
        
        X = df.drop(columns=['target']).values
        y = df['target'].values
        
        print(f"   전체 데이터: {len(df)}행 × {X.shape[1]}피처")
        print(f"   타겟 분포: {np.bincount(y)} (0:하락, 1:보합, 2:상승)")
        
        # 시계열 데이터이므로 시간 순서대로 분할
        # 1. 먼저 Test 분리 (마지막 test_size%)
        test_split_idx = int(len(X) * (1 - self.test_size))
        X_train_val = X[:test_split_idx]
        y_train_val = y[:test_split_idx]
        X_test = X[test_split_idx:]
        y_test = y[test_split_idx:]
        
        # 2. Train/Val 분리 (Train+Val 중에서 val_size%를 Validation으로)
        val_split_idx = int(len(X_train_val) * (1 - self.val_size))
        X_train = X_train_val[:val_split_idx]
        y_train = y_train_val[:val_split_idx]
        X_val = X_train_val[val_split_idx:]
        y_val = y_train_val[val_split_idx:]
        
        print(f"   Train: {len(X_train)}행, Val: {len(X_val)}행, Test: {len(X_test)}행")
        
        # 스케일링 (Train 데이터로만 fit)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test, df.columns.tolist())
    
    def train_xgboost(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """XGBoost 모델 학습"""
        print("\n🌳 XGBoost 학습 중...")
        
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Validation 예측 및 평가
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        # Test 예측 및 평가
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        self.models['xgboost'] = model
        self.results['xgboost'] = {
            'val_accuracy': val_accuracy,
            'val_f1_score': val_f1,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1,
            'val_predictions': y_val_pred,
            'val_true_labels': y_val,
            'test_predictions': y_test_pred,
            'test_true_labels': y_test
        }
        
        print(f"   Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
        return model
    
    def train_lightgbm(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """LightGBM 모델 학습"""
        print("\n💡 LightGBM 학습 중...")
        
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbose=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=['train', 'valid'],
            
        )
        
        # Validation 예측 및 평가
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        # Test 예측 및 평가
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        self.models['lightgbm'] = model
        self.results['lightgbm'] = {
            'val_accuracy': val_accuracy,
            'val_f1_score': val_f1,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1,
            'val_predictions': y_val_pred,
            'val_true_labels': y_val,
            'test_predictions': y_test_pred,
            'test_true_labels': y_test
        }
        
        print(f"   Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
        return model
    
    def train_random_forest(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Random Forest 모델 학습"""
        print("\n🌲 Random Forest 학습 중...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Validation 예측 및 평가
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred, average='weighted')
        
        # Test 예측 및 평가
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        self.models['random_forest'] = model
        self.results['random_forest'] = {
            'val_accuracy': val_accuracy,
            'val_f1_score': val_f1,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1,
            'val_predictions': y_val_pred,
            'val_true_labels': y_val,
            'test_predictions': y_test_pred,
            'test_true_labels': y_test
        }
        
        print(f"   Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
        return model
    
    def train_lstm(self, X_train, X_val, X_test, y_train, y_val, y_test, sequence_length=10):
        """LSTM 모델 학습 (시계열 데이터를 시퀀스로 변환)"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.utils import to_categorical
        except ImportError:
            print("\n⚠️ TensorFlow가 설치되지 않아 LSTM 모델을 건너뜁니다.")
            return None
        
        print(f"\n🧠 LSTM 학습 중... (시퀀스 길이: {sequence_length})")
        
        # 시계열 데이터를 시퀀스로 변환
        def create_sequences(X, y, seq_length):
            X_seq, y_seq = [], []
            for i in range(seq_length, len(X)):
                X_seq.append(X[i-seq_length:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
        
        # 원-핫 인코딩
        y_train_cat = to_categorical(y_train_seq, num_classes=3)
        y_val_cat = to_categorical(y_val_seq, num_classes=3)
        y_test_cat = to_categorical(y_test_seq, num_classes=3)
        
        print(f"   시퀀스 데이터: Train {len(X_train_seq)}개, Val {len(X_val_seq)}개, Test {len(X_test_seq)}개")
        
        # 모델 구성
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 학습
        history = model.fit(
            X_train_seq, y_train_cat,
            batch_size=32,
            epochs=50,
            validation_data=(X_val_seq, y_val_cat),
            verbose=0
        )
        
        # Validation 예측 및 평가
        y_val_pred_proba = model.predict(X_val_seq, verbose=0)
        y_val_pred = np.argmax(y_val_pred_proba, axis=1)
        val_accuracy = accuracy_score(y_val_seq, y_val_pred)
        val_f1 = f1_score(y_val_seq, y_val_pred, average='weighted')
        
        # Test 예측 및 평가
        y_test_pred_proba = model.predict(X_test_seq, verbose=0)
        y_test_pred = np.argmax(y_test_pred_proba, axis=1)
        test_accuracy = accuracy_score(y_test_seq, y_test_pred)
        test_f1 = f1_score(y_test_seq, y_test_pred, average='weighted')
        
        self.models['lstm'] = model
        self.results['lstm'] = {
            'val_accuracy': val_accuracy,
            'val_f1_score': val_f1,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1,
            'val_predictions': y_val_pred,
            'val_true_labels': y_val_seq,
            'test_predictions': y_test_pred,
            'test_true_labels': y_test_seq,
            'history': history.history
        }
        
        print(f"   Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        print(f"   Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")
        return model
    
    def plot_feature_importance(self, feature_names):
        """피처 중요도 시각화"""
        print("\n📊 피처 중요도 분석 중...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # XGBoost 피처 중요도
        if 'xgboost' in self.models:
            xgb_importance = self.models['xgboost'].feature_importances_
            indices = np.argsort(xgb_importance)[::-1][:15]
            axes[0].barh(range(len(indices)), xgb_importance[indices])
            axes[0].set_yticks(range(len(indices)))
            axes[0].set_yticklabels([feature_names[i] for i in indices])
            axes[0].set_title('XGBoost Feature Importance')
            axes[0].invert_yaxis()
        
        # LightGBM 피처 중요도
        if 'lightgbm' in self.models:
            lgbm_importance = self.models['lightgbm'].feature_importances_
            indices = np.argsort(lgbm_importance)[::-1][:15]
            axes[1].barh(range(len(indices)), lgbm_importance[indices])
            axes[1].set_yticks(range(len(indices)))
            axes[1].set_yticklabels([feature_names[i] for i in indices])
            axes[1].set_title('LightGBM Feature Importance')
            axes[1].invert_yaxis()
        
        # Random Forest 피처 중요도
        if 'random_forest' in self.models:
            rf_importance = self.models['random_forest'].feature_importances_
            indices = np.argsort(rf_importance)[::-1][:15]
            axes[2].barh(range(len(indices)), rf_importance[indices])
            axes[2].set_yticks(range(len(indices)))
            axes[2].set_yticklabels([feature_names[i] for i in indices])
            axes[2].set_title('Random Forest Feature Importance')
            axes[2].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"   저장: {self.model_dir / 'feature_importance.png'}")
        plt.close()
    
    def plot_confusion_matrices(self):
        """혼동 행렬 시각화 (Validation & Test)"""
        print("\n📈 혼동 행렬 생성 중...")
        
        n_models = len(self.results)
        if n_models == 0:
            return
        
        # Validation과 Test 각각 생성
        for dataset_type in ['val', 'test']:
            fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
            if n_models == 1:
                axes = [axes]
            
            for idx, (model_name, result) in enumerate(self.results.items()):
                true_labels = result[f'{dataset_type}_true_labels']
                predictions = result[f'{dataset_type}_predictions']
                accuracy = result[f'{dataset_type}_accuracy']
                
                cm = confusion_matrix(true_labels, predictions)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                           xticklabels=['하락(0)', '보합(1)', '상승(2)'],
                           yticklabels=['하락(0)', '보합(1)', '상승(2)'])
                axes[idx].set_title(f'{model_name.upper()} - {dataset_type.upper()}\nAccuracy: {accuracy:.4f}')
                axes[idx].set_ylabel('실제')
                axes[idx].set_xlabel('예측')
            
            plt.tight_layout()
            save_path = self.model_dir / f'confusion_matrices_{dataset_type}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   저장: {save_path}")
            plt.close()
    
    def print_comparison_report(self):
        """모델 비교 리포트 출력"""
        print("\n" + "="*60)
        print("📊 모델 성능 비교 (Validation)")
        print("="*60)
        
        # Validation 성능 비교 테이블
        val_comparison = []
        for model_name, result in self.results.items():
            val_comparison.append({
                'Model': model_name.upper(),
                'Val Accuracy': f"{result['val_accuracy']:.4f}",
                'Val F1-Score': f"{result['val_f1_score']:.4f}"
            })
        
        df_val_comparison = pd.DataFrame(val_comparison)
        print(df_val_comparison.to_string(index=False))
        
        print("\n" + "="*60)
        print("📊 모델 성능 비교 (Test)")
        print("="*60)
        
        # Test 성능 비교 테이블
        test_comparison = []
        for model_name, result in self.results.items():
            test_comparison.append({
                'Model': model_name.upper(),
                'Test Accuracy': f"{result['test_accuracy']:.4f}",
                'Test F1-Score': f"{result['test_f1_score']:.4f}"
            })
        
        df_test_comparison = pd.DataFrame(test_comparison)
        print(df_test_comparison.to_string(index=False))
        
        # 최고 성능 모델 (Test 기준)
        best_model = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\n🏆 최고 성능 모델 (Test): {best_model[0].upper()}")
        print(f"   Test Accuracy: {best_model[1]['test_accuracy']:.4f}")
        print(f"   Test F1-Score: {best_model[1]['test_f1_score']:.4f}")
        
        # CV 결과 출력
        if self.cv_results:
            print("\n" + "="*60)
            print("📊 K-Fold 교차 검증 결과")
            print("="*60)
            for model_name, cv_result in self.cv_results.items():
                print(f"\n[{model_name.upper()}]")
                print(f"   CV Mean Accuracy: {cv_result['mean_accuracy']:.4f} (±{cv_result['std_accuracy']:.4f})")
                print(f"   CV Mean F1-Score: {cv_result['mean_f1']:.4f} (±{cv_result['std_f1']:.4f})")
        
        # 상세 리포트 (Validation)
        print("\n" + "="*60)
        print("📋 상세 분류 리포트 (Validation)")
        print("="*60)
        
        for model_name, result in self.results.items():
            print(f"\n[{model_name.upper()}]")
            print(classification_report(
                result['val_true_labels'],
                result['val_predictions'],
                target_names=['하락(0)', '보합(1)', '상승(2)']
            ))
        
        # 상세 리포트 (Test)
        print("\n" + "="*60)
        print("📋 상세 분류 리포트 (Test)")
        print("="*60)
        
        for model_name, result in self.results.items():
            print(f"\n[{model_name.upper()}]")
            print(classification_report(
                result['test_true_labels'],
                result['test_predictions'],
                target_names=['하락(0)', '보합(1)', '상승(2)']
            ))
    
    def save_models(self):
        """모델 저장"""
        print("\n💾 모델 저장 중...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 스케일러 저장
        scaler_path = self.model_dir / f'scaler_{timestamp}.pkl'
        joblib.dump(self.scalers['standard'], scaler_path)
        print(f"   스케일러: {scaler_path}")
        
        # 각 모델 저장
        for model_name, model in self.models.items():
            if model_name == 'lstm':
                # LSTM은 Keras 모델
                model_path = self.model_dir / f'{model_name}_{timestamp}.h5'
                model.save(model_path)
            else:
                # 트리 기반 모델은 joblib로 저장
                model_path = self.model_dir / f'{model_name}_{timestamp}.pkl'
                joblib.dump(model, model_path)
            
            print(f"   {model_name}: {model_path}")
        
        # 결과 저장
        results_summary = {}
        for model_name, result in self.results.items():
            results_summary[model_name] = {
                'val_accuracy': float(result['val_accuracy']),
                'val_f1_score': float(result['val_f1_score']),
                'test_accuracy': float(result['test_accuracy']),
                'test_f1_score': float(result['test_f1_score'])
            }
        
        # CV 결과 추가
        if self.cv_results:
            for model_name, cv_result in self.cv_results.items():
                if model_name not in results_summary:
                    results_summary[model_name] = {}
                results_summary[model_name]['cv_mean_accuracy'] = float(cv_result['mean_accuracy'])
                results_summary[model_name]['cv_std_accuracy'] = float(cv_result['std_accuracy'])
                results_summary[model_name]['cv_mean_f1'] = float(cv_result['mean_f1'])
                results_summary[model_name]['cv_std_f1'] = float(cv_result['std_f1'])
        
        results_path = self.model_dir / f'results_{timestamp}.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        print(f"   결과: {results_path}")
    
    def cross_validate(self, X_train_val, y_train_val, feature_names, use_lstm=True):
        """K-Fold 교차 검증"""
        print("\n" + "="*60)
        print(f"🔄 K-Fold 교차 검증 시작 (n_splits={self.n_splits})")
        print("="*60)
        
        cv_scores = {model_name: {'accuracy': [], 'f1': []} 
                     for model_name in ['xgboost', 'lightgbm', 'random_forest']}
        if use_lstm:
            cv_scores['lstm'] = {'accuracy': [], 'f1': []}
        
        fold = 1
        for train_idx, val_idx in self.tscv.split(X_train_val):
            print(f"\n📁 Fold {fold}/{self.n_splits}")
            print(f"   Train: {len(train_idx)}행, Val: {len(val_idx)}행")
            
            X_train_fold = X_train_val[train_idx]
            X_val_fold = X_train_val[val_idx]
            y_train_fold = y_train_val[train_idx]
            y_val_fold = y_train_val[val_idx]
            
            # 각 모델별로 교차 검증
            # XGBoost
            model_xgb = xgb.XGBClassifier(
                objective='multi:softprob', num_class=3,
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, eval_metric='mlogloss',
                use_label_encoder=False
            )
            model_xgb.fit(X_train_fold, y_train_fold, verbose=False)
            y_pred_xgb = model_xgb.predict(X_val_fold)
            cv_scores['xgboost']['accuracy'].append(accuracy_score(y_val_fold, y_pred_xgb))
            cv_scores['xgboost']['f1'].append(f1_score(y_val_fold, y_pred_xgb, average='weighted'))
            
            # LightGBM
            model_lgb = lgb.LGBMClassifier(
                objective='multiclass', num_class=3,
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, verbose=-1
            )
            model_lgb.fit(X_train_fold, y_train_fold)
            y_pred_lgb = model_lgb.predict(X_val_fold)
            cv_scores['lightgbm']['accuracy'].append(accuracy_score(y_val_fold, y_pred_lgb))
            cv_scores['lightgbm']['f1'].append(f1_score(y_val_fold, y_pred_lgb, average='weighted'))
            
            # Random Forest
            model_rf = RandomForestClassifier(
                n_estimators=200, max_depth=10,
                min_samples_split=5, min_samples_leaf=2,
                random_state=self.random_state, n_jobs=-1
            )
            model_rf.fit(X_train_fold, y_train_fold)
            y_pred_rf = model_rf.predict(X_val_fold)
            cv_scores['random_forest']['accuracy'].append(accuracy_score(y_val_fold, y_pred_rf))
            cv_scores['random_forest']['f1'].append(f1_score(y_val_fold, y_pred_rf, average='weighted'))
            
            # LSTM
            if use_lstm:
                try:
                    from tensorflow.keras.models import Sequential
                    from tensorflow.keras.layers import LSTM, Dense, Dropout
                    from tensorflow.keras.utils import to_categorical
                    
                    def create_sequences(X, y, seq_length=10):
                        X_seq, y_seq = [], []
                        for i in range(seq_length, len(X)):
                            X_seq.append(X[i-seq_length:i])
                            y_seq.append(y[i])
                        return np.array(X_seq), np.array(y_seq)
                    
                    X_train_seq, y_train_seq = create_sequences(X_train_fold, y_train_fold)
                    X_val_seq, y_val_seq = create_sequences(X_val_fold, y_val_fold)
                    
                    if len(X_train_seq) > 0 and len(X_val_seq) > 0:
                        y_train_cat = to_categorical(y_train_seq, num_classes=3)
                        y_val_cat = to_categorical(y_val_seq, num_classes=3)
                        
                        model_lstm = Sequential([
                            LSTM(64, return_sequences=True, input_shape=(10, X_train_fold.shape[1])),
                            Dropout(0.2),
                            LSTM(32, return_sequences=False),
                            Dropout(0.2),
                            Dense(16, activation='relu'),
                            Dense(3, activation='softmax')
                        ])
                        model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                        model_lstm.fit(X_train_seq, y_train_cat, batch_size=32, epochs=20, verbose=0)
                        
                        y_pred_lstm = np.argmax(model_lstm.predict(X_val_seq, verbose=0), axis=1)
                        cv_scores['lstm']['accuracy'].append(accuracy_score(y_val_seq, y_pred_lstm))
                        cv_scores['lstm']['f1'].append(f1_score(y_val_seq, y_pred_lstm, average='weighted'))
                except:
                    pass
            
            fold += 1
        
        # CV 결과 요약
        for model_name, scores in cv_scores.items():
            if len(scores['accuracy']) > 0:
                self.cv_results[model_name] = {
                    'mean_accuracy': np.mean(scores['accuracy']),
                    'std_accuracy': np.std(scores['accuracy']),
                    'mean_f1': np.mean(scores['f1']),
                    'std_f1': np.std(scores['f1'])
                }
        
        print("\n✅ 교차 검증 완료!")
    
    def train_all(self, use_lstm=True):
        """모든 모델 학습"""
        print("="*60)
        print("🚀 주식 예측 모델 학습 시작")
        print("="*60)
        
        # 데이터 로드 (Train/Val/Test 분할)
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = self.load_data()
        
        # K-Fold 교차 검증 (Train+Val 데이터 사용)
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.hstack([y_train, y_val])
        self.cross_validate(X_train_val, y_train_val, feature_names, use_lstm)
        
        # 최종 모델 학습 (전체 Train 데이터로 학습, Val로 검증, Test로 최종 평가)
        print("\n" + "="*60)
        print("🎯 최종 모델 학습")
        print("="*60)
        
        self.train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test)
        self.train_lightgbm(X_train, X_val, X_test, y_train, y_val, y_test)
        self.train_random_forest(X_train, X_val, X_test, y_train, y_val, y_test)
        
        if use_lstm:
            self.train_lstm(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # 시각화 및 리포트
        self.plot_feature_importance(feature_names)
        self.plot_confusion_matrices()
        self.print_comparison_report()
        
        # 모델 저장
        self.save_models()
        
        print("\n" + "="*60)
        print("✅ 학습 완료!")
        print("="*60)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="주식 예측 모델 학습")
    parser.add_argument("--data", type=str, 
                       default="D:/stock/_v7/_data/preprocessed_005930_20100101_20260129.csv",
                       help="전처리된 데이터 파일 경로")
    parser.add_argument("--model-dir", type=str, default="D:/stock/_v7/models",
                       help="모델 저장 디렉토리")
    parser.add_argument("--no-lstm", action="store_true",
                       help="LSTM 모델 제외")
    parser.add_argument("--n-splits", type=int, default=5,
                       help="K-Fold 교차 검증 fold 수")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="테스트 데이터 비율")
    parser.add_argument("--val-size", type=float, default=0.2,
                       help="Validation 데이터 비율 (Train 데이터 기준)")
    
    args = parser.parse_args()
    
    trainer = StockModelTrainer(
        data_path=args.data,
        model_dir=args.model_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        n_splits=args.n_splits
    )
    
    trainer.train_all(use_lstm=not args.no_lstm)

