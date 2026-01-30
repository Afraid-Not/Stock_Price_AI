import pandas as pd
import numpy as np
import argparse
import os
import joblib
import json
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
import optuna
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 영어 폰트 설정

class StockEnsembleTrainer:
    def __init__(self, data_path, model_dir="D:/stock/_v6/models", scaler_dir="D:/stock/_v6/scalers", n_splits=5, stock_code=None, use_optuna=True, n_trials=50):
        self.data_path = data_path
        self.base_model_dir = Path(model_dir)
        self.base_model_dir.mkdir(parents=True, exist_ok=True)
        self.scaler_dir = Path(scaler_dir)
        self.n_splits = n_splits
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.stock_code = stock_code
        self.use_optuna = use_optuna
        self.n_trials = n_trials
        
        # 종목코드별 모델 디렉토리 생성
        if stock_code:
            self.model_dir = self.base_model_dir / stock_code
            self.model_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.model_dir = self.base_model_dir
        
        # Target scaler는 나중에 로드 (종목코드가 확정된 후)
        self.target_scaler = None
        
        # 최적 파라미터 저장 경로 (종목코드별 폴더 내에 저장)
        # 종목코드가 있으면 종목코드 폴더에, 없으면 기본 폴더에 저장
        if stock_code and stock_code != "UNKNOWN":
            self.params_dir = self.model_dir  # 종목코드 폴더에 직접 저장
        else:
            self.params_dir = self.base_model_dir / "params"
            self.params_dir.mkdir(parents=True, exist_ok=True)
        
        # 최적 파라미터 (로드되거나 최적화 후 설정됨)
        self.best_lgbm_params = None
    
    def _load_target_scaler(self):
        """Target scaler 로드 (종목별)"""
        if self.stock_code:
            target_scaler_path = self.scaler_dir / f"{self.stock_code}_target_scaler.bin"
            if target_scaler_path.exists():
                self.target_scaler = joblib.load(target_scaler_path)
                print(f"✅ Target scaler loaded: {target_scaler_path}")
            else:
                # 종목별 스케일러가 없으면 DEFAULT 스케일러 시도
                default_scaler_path = self.scaler_dir / "DEFAULT_target_scaler.bin"
                if default_scaler_path.exists():
                    self.target_scaler = joblib.load(default_scaler_path)
                    print(f"⚠️ Using DEFAULT scaler: {default_scaler_path}")
                else:
                    print(f"⚠️ Target scaler not found: {target_scaler_path}")
                    print(f"   종목별 스케일러가 없습니다. 테스트 예측 시 스케일링된 값을 사용합니다.")
                    self.target_scaler = None
        else:
            # 종목코드가 없으면 기본 스케일러 시도
            target_scaler_path = self.scaler_dir / "target_scaler.bin"
            if target_scaler_path.exists():
                self.target_scaler = joblib.load(target_scaler_path)
                print(f"✅ Target scaler loaded: {target_scaler_path}")
            else:
                print(f"⚠️ Target scaler not found: {target_scaler_path}")
                self.target_scaler = None
        
    def load_data(self, test_size=50):
        """데이터 로드 및 분리 (마지막 test_size행을 test set으로)"""
        print(f"📂 데이터 로드: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # Target과 Feature 분리
        if 'target' not in df.columns:
            raise ValueError("'target' 컬럼이 없습니다.")
        
        # 마지막 test_size행을 test set으로 분리
        if len(df) <= test_size:
            raise ValueError(f"데이터가 너무 적습니다. 최소 {test_size + 1}행 필요합니다.")
        
        df_train_val = df.iloc[:-test_size].copy()
        df_test = df.iloc[-test_size:].copy()
        
        X_train_val = df_train_val.drop(columns=['target']).values
        y_train_val = df_train_val['target'].values
        
        X_test = df_test.drop(columns=['target']).values
        y_test = df_test['target'].values
        
        print(f"   전체 데이터: {len(df)}행 × {X_train_val.shape[1]}피처")
        print(f"   Train+Val: {len(df_train_val)}행")
        print(f"   Test: {len(df_test)}행")
        print(f"   Target 범위: [{y_train_val.min():.4f}, {y_train_val.max():.4f}]")
        
        return X_train_val, y_train_val, X_test, y_test
    
    def _cleanup_old_models(self, stock_code, current_model_name):
        """해당 종목의 이전 모델 파일들 삭제 (현재 모델 제외)"""
        try:
            # 종목코드 폴더 내의 모든 모델 파일 찾기
            lgbm_files = list(self.model_dir.glob("*_lgbm.txt"))
            weights_files = list(self.model_dir.glob("*_weights.json"))
            graph_files = list(self.model_dir.glob("*_test_prediction*.png"))
            
            deleted_count = 0
            for file_path in lgbm_files + weights_files + graph_files:
                # 현재 모델 파일명과 일치하지 않으면 삭제
                if current_model_name not in file_path.stem:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        print(f"   ⚠️ 파일 삭제 실패: {file_path.name} - {e}")
            
            if deleted_count > 0:
                print(f"   🗑️ 이전 모델 {deleted_count}개 삭제됨")
        except Exception as e:
            print(f"   ⚠️ 모델 정리 중 오류: {e}")
    
    def _load_best_params(self):
        """저장된 최적 파라미터 로드"""
        if not self.stock_code:
            return None
        
        # 종목코드 폴더에 있으면 파일명에서 종목코드 제거, 아니면 종목코드 포함
        if self.params_dir == self.model_dir and self.stock_code != "UNKNOWN":
            # 종목코드 폴더에 직접 저장된 경우
            lgbm_params_path = self.params_dir / "lgbm_params.json"
        else:
            # 기본 params 폴더에 저장된 경우 (하위 호환성)
            lgbm_params_path = self.params_dir / f"{self.stock_code}_lgbm_params.json"
        
        lgbm_params = None
        
        if lgbm_params_path.exists():
            try:
                with open(lgbm_params_path, 'r') as f:
                    lgbm_params = json.load(f)
                print(f"✅ LGBM 최적 파라미터 로드: {lgbm_params_path}")
            except Exception as e:
                print(f"⚠️ LGBM 파라미터 로드 실패: {e}")
        
        return lgbm_params
    
    def _save_best_params(self, lgbm_params):
        """최적 파라미터 저장"""
        if not self.stock_code:
            return
        
        # 종목코드 폴더에 있으면 파일명에서 종목코드 제거, 아니면 종목코드 포함
        if self.params_dir == self.model_dir and self.stock_code != "UNKNOWN":
            # 종목코드 폴더에 직접 저장
            lgbm_params_path = self.params_dir / "lgbm_params.json"
        else:
            # 기본 params 폴더에 저장 (하위 호환성)
            lgbm_params_path = self.params_dir / f"{self.stock_code}_lgbm_params.json"
        
        try:
            with open(lgbm_params_path, 'w') as f:
                json.dump(lgbm_params, f, indent=2)
            print(f"💾 LGBM 최적 파라미터 저장: {lgbm_params_path}")
        except Exception as e:
            print(f"⚠️ LGBM 파라미터 저장 실패: {e}")
    
    def _optimize_lgbm(self, X_train_val, y_train_val):
        """Optuna로 LightGBM 하이퍼파라미터 최적화"""
        print(f"\n🔍 Optuna로 LightGBM 하이퍼파라미터 최적화 시작 (n_trials={self.n_trials})...")
        
        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'verbose': -1,
                'seed': 42
            }
            
            # KFold로 검증
            kf = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in kf.split(X_train_val):
                X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
                y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[val_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )
                
                pred = model.predict(X_val, num_iteration=model.best_iteration)
                rmse = np.sqrt(mean_squared_error(y_val, pred))
                scores.append(rmse)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        best_params = study.best_params.copy()
        best_params['objective'] = 'regression'
        best_params['metric'] = 'rmse'
        best_params['boosting_type'] = 'gbdt'
        best_params['verbose'] = -1
        best_params['seed'] = 42
        
        print(f"✅ LGBM 최적 파라미터 찾음 (RMSE: {study.best_value:.6f})")
        return best_params
    
    def train_lgbm(self, X_train, y_train, X_val, y_val, fold_idx):
        """LightGBM 모델 학습"""
        print(f"\n🌲 LightGBM 학습 중...")
        
        # 회귀 문제이므로 objective='regression' 사용
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 최적 파라미터가 있으면 사용, 없으면 기본값
        if self.best_lgbm_params:
            params = self.best_lgbm_params.copy()
            print(f"   ✅ 최적 파라미터 사용")
        else:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42
            }
            print(f"   ⚠️ 기본 파라미터 사용")
        
        # 얼리스타핑 콜백
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'eval'],
            callbacks=callbacks
        )
        
        return model
    
    def evaluate_regression(self, y_true, y_pred):
        """회귀 모델 평가 지표"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # F1 스코어를 위해 이진 분류로 변환 (양수=1, 음수=0)
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        f1 = f1_score(y_true_binary, y_pred_binary, average='binary')
        acc = accuracy_score(y_true_binary, y_pred_binary)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'f1': f1,
            'acc': acc
        }
    
    def plot_test_results(self, y_true, y_pred, metrics, save_path):
        """Test 예측 결과 그래프 생성 및 저장"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        n = len(y_true)
        x = np.arange(n)
        
        # 1. Actual vs Predicted (Line plot)
        axes[0, 0].plot(x, y_true, 'o-', label='Actual', linewidth=2, markersize=6, color='blue')
        axes[0, 0].plot(x, y_pred, 's-', label='Predicted', linewidth=2, markersize=6, color='red', alpha=0.7)
        axes[0, 0].set_xlabel('Sample Index', fontsize=12)
        axes[0, 0].set_ylabel('Return Rate', fontsize=12)
        axes[0, 0].set_title('Actual vs Predicted Returns', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=11)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Scatter plot
        axes[0, 1].scatter(y_true, y_pred, alpha=0.6, s=50)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[0, 1].set_xlabel('Actual Return Rate', fontsize=12)
        axes[0, 1].set_ylabel('Predicted Return Rate', fontsize=12)
        axes[0, 1].set_title('Prediction Scatter Plot', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals (Error)
        residuals = y_true - y_pred
        axes[1, 0].plot(x, residuals, 'o-', linewidth=2, markersize=6, color='green', alpha=0.7)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Sample Index', fontsize=12)
        axes[1, 0].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
        axes[1, 0].set_title('Prediction Residuals', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Metrics summary
        axes[1, 1].axis('off')
        metrics_text = f"""
Test Set Performance Metrics:

Regression Metrics:
  RMSE: {metrics['rmse']:.6f}
  MAE:  {metrics['mae']:.6f}
  MSE:  {metrics['mse']:.6f}

Classification Metrics (Binary):
  Accuracy: {metrics['acc']:.4f}
  F1 Score: {metrics['f1']:.4f}

Sample Size: {n}
        """
        axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Test prediction graph saved: {save_path}")
        plt.close()
    
    def train_ensemble(self):
        """앙상블 모델 학습 (KFold) 및 Test 예측"""
        X_train_val, y_train_val, X_test, y_test = self.load_data()
        
        # 최적 파라미터 로드 또는 최적화
        if self.use_optuna:
            # 저장된 파라미터가 있는지 확인
            lgbm_params = self._load_best_params()
            
            if lgbm_params is None:
                print(f"\n{'='*60}")
                print(f"🔍 하이퍼파라미터 최적화 시작")
                print(f"{'='*60}")
                
                # Optuna로 최적화
                lgbm_params = self._optimize_lgbm(X_train_val, y_train_val)
                
                # 최적 파라미터 저장
                self._save_best_params(lgbm_params)
            else:
                print(f"\n{'='*60}")
                print(f"📂 저장된 최적 파라미터 사용")
                print(f"{'='*60}")
            
            self.best_lgbm_params = lgbm_params
        else:
            # Optuna 사용 안 함 - 기본 파라미터 사용
            print(f"\n{'='*60}")
            print(f"⚠️ Optuna 최적화 비활성화 - 기본 파라미터 사용")
            print(f"{'='*60}")
            self.best_lgbm_params = None
        
        fold_results = []
        best_f1 = -1
        best_fold = -1
        best_lgbm_model = None
        best_ensemble_weights = None  # 최적 앙상블 가중치 저장
        
        print(f"\n{'='*60}")
        print(f"🎯 KFold 교차 검증 시작 (n_splits={self.n_splits})")
        print(f"{'='*60}")
        
        # 전체 fold에서 앙상블 가중치 최적화를 위한 데이터 수집
        all_lgbm_preds = []
        all_y_true = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(X_train_val)):
            print(f"\n📊 Fold {fold_idx + 1}/{self.n_splits}")
            print(f"   Train: {len(train_idx)}개, Val: {len(val_idx)}개")
            
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
            
            # 모델 학습
            lgbm_model = self.train_lgbm(X_train, y_train, X_val, y_val, fold_idx)
            
            # 예측
            lgbm_pred = lgbm_model.predict(X_val, num_iteration=lgbm_model.best_iteration)
            
            # 모델 성능 평가
            lgbm_metrics = self.evaluate_regression(y_val, lgbm_pred)
            
            # LGBM만 사용하므로 가중치는 1.0
            lgbm_weight = 1.0
            
            print(f"\n   모델별 성능:")
            print(f"   LGBM: F1={lgbm_metrics['f1']:.4f}, RMSE={lgbm_metrics['rmse']:.6f} → 가중치: {lgbm_weight:.3f}")
            
            # 앙상블 예측 (LGBM만 사용)
            ensemble_pred = lgbm_pred
            
            # 전체 fold 데이터 수집
            all_lgbm_preds.append(lgbm_pred)
            all_y_true.append(y_val)
            
            # 평가
            metrics = self.evaluate_regression(y_val, ensemble_pred)
            
            print(f"\n✅ Fold {fold_idx + 1} 결과:")
            print(f"   RMSE: {metrics['rmse']:.6f}")
            print(f"   MAE:  {metrics['mae']:.6f}")
            print(f"   F1:   {metrics['f1']:.4f}")
            print(f"   Acc:  {metrics['acc']:.4f}")
            
            fold_results.append({
                'fold': fold_idx + 1,
                'metrics': metrics,
                'lgbm_model': lgbm_model,
                'lgbm_weight': lgbm_weight
            })
            
            # F1 스코어 기준으로 최고 모델 저장
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_fold = fold_idx + 1
                best_lgbm_model = lgbm_model
                best_ensemble_weights = {
                    'lgbm': lgbm_weight
                }
                
                # 모델 저장 (종목코드 포함)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                stock_code = getattr(self, 'stock_code', 'UNKNOWN')
                model_name = f"{stock_code}_fold{best_fold}_acc{metrics['acc']:.4f}_f1{metrics['f1']:.4f}_{timestamp}"
                
                lgbm_path = self.model_dir / f"{model_name}_lgbm.txt"
                
                # 이전 모델 삭제 (해당 종목의 이전 모델들만)
                if stock_code != 'UNKNOWN':
                    self._cleanup_old_models(stock_code, model_name)
                
                lgbm_model.save_model(str(lgbm_path))
                
                # 앙상블 가중치 저장
                weights_path = self.model_dir / f"{model_name}_weights.json"
                with open(weights_path, 'w') as f:
                    json.dump(best_ensemble_weights, f, indent=2)
                
                print(f"\n💾 최고 모델 저장: {model_name}")
                print(f"   LGBM: {lgbm_path}")
                weight_str = f"LGBM={best_ensemble_weights['lgbm']:.3f}"
                print(f"   가중치: {weight_str}")
        
        # 전체 결과 요약
        print(f"\n{'='*60}")
        print(f"📈 전체 결과 요약")
        print(f"{'='*60}")
        
        avg_metrics = {
            'rmse': np.mean([r['metrics']['rmse'] for r in fold_results]),
            'mae': np.mean([r['metrics']['mae'] for r in fold_results]),
            'f1': np.mean([r['metrics']['f1'] for r in fold_results]),
            'acc': np.mean([r['metrics']['acc'] for r in fold_results])
        }
        
        std_metrics = {
            'rmse': np.std([r['metrics']['rmse'] for r in fold_results]),
            'mae': np.std([r['metrics']['mae'] for r in fold_results]),
            'f1': np.std([r['metrics']['f1'] for r in fold_results]),
            'acc': np.std([r['metrics']['acc'] for r in fold_results])
        }
        
        print(f"평균 ± 표준편차:")
        print(f"  RMSE: {avg_metrics['rmse']:.6f} ± {std_metrics['rmse']:.6f}")
        print(f"  MAE:  {avg_metrics['mae']:.6f} ± {std_metrics['mae']:.6f}")
        print(f"  F1:   {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")
        print(f"  Acc:  {avg_metrics['acc']:.4f} ± {std_metrics['acc']:.4f}")
        print(f"\n최고 F1 스코어: {best_f1:.4f} (Fold {best_fold})")
        
        # 전체 fold 데이터로 최적 앙상블 가중치 찾기 (LGBM만 사용)
        if len(all_lgbm_preds) > 0:
            all_lgbm_pred = np.concatenate(all_lgbm_preds)
            all_y = np.concatenate(all_y_true)
            
            # LGBM만 사용하므로 가중치는 1.0
            optimal_lgbm_weight = 1.0
            
            ensemble_pred = optimal_lgbm_weight * all_lgbm_pred
            metrics = self.evaluate_regression(all_y, ensemble_pred)
            best_weight_f1 = metrics['f1']
            
            print(f"\n✅ 최적 앙상블 가중치:")
            print(f"   LGBM: {optimal_lgbm_weight:.3f} (F1={best_weight_f1:.4f})")
            
            # 최적 가중치를 best_ensemble_weights로 업데이트
            if best_ensemble_weights is None:
                best_ensemble_weights = {
                    'lgbm': optimal_lgbm_weight
                }
            else:
                current_ensemble = best_ensemble_weights['lgbm'] * all_lgbm_pred
                current_metrics = self.evaluate_regression(all_y, current_ensemble)
                if best_weight_f1 > current_metrics['f1']:
                    best_ensemble_weights = {
                        'lgbm': optimal_lgbm_weight
                    }
                    print(f"   ⚡ 최적 가중치로 업데이트됨 (F1: {current_metrics['f1']:.4f} → {best_weight_f1:.4f})")
        
        # Test set 예측
        if best_lgbm_model is not None:
            print(f"\n{'='*60}")
            print(f"🧪 Test Set Prediction (Last 50 samples)")
            print(f"{'='*60}")
            
            # 전체 train_val 데이터로 재학습 (최고 모델의 하이퍼파라미터 사용)
            print("\n🔄 Retraining on full train+val data for test prediction...")
            final_lgbm = self.train_lgbm(X_train_val, y_train_val, X_test, y_test, -1)
            
            # Test 예측
            lgbm_test_pred = final_lgbm.predict(X_test, num_iteration=final_lgbm.best_iteration)
            
            # 최적 가중치로 앙상블 예측 (LGBM만 사용)
            if best_ensemble_weights:
                ensemble_test_pred = best_ensemble_weights['lgbm'] * lgbm_test_pred
                weight_str = f"LGBM={best_ensemble_weights['lgbm']:.3f}"
                print(f"\n   앙상블 가중치 적용: {weight_str}")
            else:
                ensemble_test_pred = lgbm_test_pred
                print(f"\n   기본 가중치 사용 (LGBM만)")
            
            # 스케일러 역변환 (원본 수익률로 복원)
            if self.target_scaler is not None:
                y_test_original = self.target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                ensemble_test_pred_original = self.target_scaler.inverse_transform(ensemble_test_pred.reshape(-1, 1)).flatten()
                print(f"\n🔄 Scaler inverse transform applied")
                print(f"   Scaled range: [{y_test.min():.4f}, {y_test.max():.4f}]")
                print(f"   Original range: [{y_test_original.min():.4f}, {y_test_original.max():.4f}]")
            else:
                y_test_original = y_test
                ensemble_test_pred_original = ensemble_test_pred
                print(f"\n⚠️ Scaler not found, using scaled values")
            
            # Test 평가 (원본 값으로)
            test_metrics = self.evaluate_regression(y_test_original, ensemble_test_pred_original)
            
            print(f"\n✅ Test Set Results (Original Scale):")
            print(f"   RMSE: {test_metrics['rmse']:.6f}")
            print(f"   MAE:  {test_metrics['mae']:.6f}")
            print(f"   F1:   {test_metrics['f1']:.4f}")
            print(f"   Acc:  {test_metrics['acc']:.4f}")
            
            # 그래프 저장 (원본 값으로)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stock_code = getattr(self, 'stock_code', 'UNKNOWN')
            graph_path = self.model_dir / f"{stock_code}_test_prediction_acc{test_metrics['acc']:.4f}_f1{test_metrics['f1']:.4f}_{timestamp}.png"
            self.plot_test_results(y_test_original, ensemble_test_pred_original, test_metrics, graph_path)
        
        return fold_results

def main():
    parser = argparse.ArgumentParser(description="주식 데이터 앙상블 모델 학습")
    parser.add_argument("--data", type=str, required=True, help="전처리된 데이터 경로")
    parser.add_argument("--code", type=str, default=None, help="종목코드 (예: 005930, 파일명에서 자동 추출 시도)")
    parser.add_argument("--model_dir", type=str, default="D:/stock/_v6/models", help="모델 저장 디렉토리")
    parser.add_argument("--scaler_dir", type=str, default="D:/stock/_v6/scalers", help="스케일러 디렉토리")
    parser.add_argument("--n_splits", type=int, default=5, help="KFold 분할 수")
    parser.add_argument("--use_optuna", action="store_true", default=True, help="Optuna 하이퍼파라미터 최적화 사용 (기본값: True)")
    parser.add_argument("--no_optuna", dest="use_optuna", action="store_false", help="Optuna 최적화 비활성화")
    parser.add_argument("--n_trials", type=int, default=50, help="Optuna 최적화 시도 횟수 (기본값: 50)")
    
    args = parser.parse_args()
    
    # 종목코드 추출 (파일명에서 또는 직접 입력)
    stock_code = args.code
    if stock_code is None:
        # 파일명에서 추출 시도 (예: preprocessed_005930_20260101_20260127.csv)
        import re
        filename = Path(args.data).stem
        match = re.search(r'(\d{6})', filename)
        if match:
            stock_code = match.group(1)
            print(f"📌 종목코드 자동 추출: {stock_code}")
        else:
            stock_code = "UNKNOWN"
            print(f"⚠️ 종목코드를 찾을 수 없습니다. 'UNKNOWN'으로 저장됩니다.")
            print(f"   --code 옵션으로 직접 지정하세요.")
    
    print(f"\n{'='*60}")
    print(f"🚀 Training Started")
    print(f"{'='*60}")
    print(f"Stock Code: {stock_code}")
    print(f"Data: {args.data}")
    print(f"Optuna: {args.use_optuna} (n_trials={args.n_trials})")
    print(f"{'='*60}\n")
    
    trainer = StockEnsembleTrainer(
        data_path=args.data,
        model_dir=args.model_dir,
        scaler_dir=args.scaler_dir,
        n_splits=args.n_splits,
        stock_code=stock_code,
        use_optuna=args.use_optuna,
        n_trials=args.n_trials
    )
    
    # 종목코드가 설정된 후 모델 디렉토리 재설정
    if stock_code and stock_code != "UNKNOWN":
        trainer.model_dir = trainer.base_model_dir / stock_code
        trainer.model_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 모델 저장 경로: {trainer.model_dir}")
    
    # 스케일러 다시 로드 (종목코드가 설정된 후)
    trainer._load_target_scaler()
    
    trainer.train_ensemble()

if __name__ == "__main__":
    main()

