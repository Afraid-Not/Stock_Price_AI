import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import KFold
import joblib
import warnings
import xgboost as xgb
import lightgbm as lgb

# LightGBM feature names 경고 무시
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')

# LightGBM feature names 경고 무시
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')

def load_data(data_path):
    """데이터 로드 및 윈도우 데이터 생성"""
    df = pd.read_csv(data_path)
    
    # Target과 Feature 분리
    if 'target' not in df.columns:
        raise ValueError("데이터에 'target' 컬럼이 없습니다.")
    
    targets = df['target'].values
    features_df = df.drop(columns=['target'])
    
    # 날짜 컬럼 제거 (있다면)
    if '날짜' in features_df.columns:
        features_df = features_df.drop(columns=['날짜'])
    
    window_size = 60
    
    # 슬라이딩 윈도우로 데이터 생성
    X_list = []
    y_list = []
    
    for i in range(len(features_df) - window_size):
        window_data = features_df.iloc[i:i+window_size].values.flatten()  # (60, features) -> (60*features,)
        X_list.append(window_data)
        y_list.append(targets[i + window_size])
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    return X, y, window_size * features_df.shape[1]

class EnsembleModel:
    """XGBoost + LightGBM 앙상블 모델"""
    def __init__(self, input_dim, xgb_params=None, lgbm_params=None):
        self.input_dim = input_dim
        
        # XGBoost 기본 파라미터
        if xgb_params is None:
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'tree_method': 'hist'
            }
        
        # LightGBM 기본 파라미터
        if lgbm_params is None:
            lgbm_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
        
        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        self.lgbm_model = lgb.LGBMClassifier(**lgbm_params)
        self.ensemble_weights = [0.5, 0.5]  # XGBoost, LightGBM 가중치
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            xgb_early_stopping_rounds=10, lgbm_early_stopping_rounds=10):
        """모델 학습"""
        # XGBoost 학습
        print("🌳 Training XGBoost...")
        if X_val is not None and y_val is not None:
            # eval_set만 사용 (early stopping은 모델 파라미터에서 처리)
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.xgb_model.fit(X_train, y_train, verbose=False)
        
        # LightGBM 학습
        print("🌲 Training LightGBM...")
        if X_val is not None and y_val is not None:
            # LightGBM early stopping 시도
            try:
                # 최신 LightGBM은 callbacks 사용
                self.lgbm_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(lgbm_early_stopping_rounds), lgb.log_evaluation(0)]
                )
            except (TypeError, AttributeError):
                # callbacks 미지원 시 early_stopping_rounds 시도
                try:
                    self.lgbm_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=lgbm_early_stopping_rounds
                    )
                except TypeError:
                    # early_stopping_rounds도 지원 안하는 경우 - eval_set만 사용
                    self.lgbm_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)]
                    )
        else:
            self.lgbm_model.fit(X_train, y_train)
    
    def predict_proba(self, X):
        """앙상블 확률 예측"""
        # numpy array를 DataFrame으로 변환하여 feature names 경고 방지
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X
        
        xgb_proba = self.xgb_model.predict_proba(X_df)
        lgbm_proba = self.lgbm_model.predict_proba(X_df)
        
        # 가중 평균
        ensemble_proba = (xgb_proba * self.ensemble_weights[0] + 
                         lgbm_proba * self.ensemble_weights[1])
        
        return ensemble_proba
    
    def predict(self, X):
        """앙상블 클래스 예측"""
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

def visualize_predictions(model, data_path, num_samples=30, model_dir=None):
    """마지막 N개 데이터에 대한 예측 결과 시각화"""
    print(f"\n{'='*60}")
    print(f"📊 Visualization (Last {num_samples} samples)")
    print(f"{'='*60}")
    
    # 데이터 로드
    X, y, input_dim = load_data(data_path)
    
    window_size = 60
    if len(X) < num_samples:
        num_samples = len(X)
        print(f"⚠️ Insufficient data, visualizing {num_samples} samples.")
    
    # 마지막 num_samples개 예측
    X_test = X[-num_samples:]
    y_test = y[-num_samples:] if y is not None else None
    
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)
    confidences = proba.max(axis=1)
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. Prediction vs Actual
    x = range(len(predictions))
    axes[0].plot(x, predictions, 'o-', label='Prediction', color='blue', linewidth=2, markersize=6)
    if y_test is not None:
        axes[0].plot(x, y_test, 's-', label='Actual', color='red', linewidth=2, markersize=6)
        accuracy = accuracy_score(y_test, predictions) * 100
        axes[0].set_title(f'Prediction vs Actual (Accuracy: {accuracy:.2f}%)', fontsize=14, fontweight='bold')
    else:
        axes[0].set_title('Prediction Results', fontsize=14, fontweight='bold')
    
    axes[0].set_xlabel('Data Index', fontsize=12)
    axes[0].set_ylabel('Class (0: Down, 1: Up)', fontsize=12)
    axes[0].set_ylim([-0.1, 1.1])
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['Down', 'Up'])
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Confidence
    colors = ['red' if p == 0 else 'green' for p in predictions]
    axes[1].bar(x, confidences, color=colors, alpha=0.6, edgecolor='black', linewidth=1)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='50% Threshold')
    axes[1].set_title('Prediction Confidence', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Data Index', fontsize=12)
    axes[1].set_ylabel('Confidence', fontsize=12)
    axes[1].set_ylim([0, 1])
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 그래프 저장
    if model_dir is None:
        model_dir = Path("D:/stock/_v5/models")
    else:
        model_dir = Path(model_dir)
    
    graph_path = model_dir / f"prediction_graph_v2_{num_samples}samples.png"
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graph saved: {graph_path}")
    
    # Statistics
    print(f"\n📊 Prediction Statistics:")
    print(f"   - Mean Confidence: {np.mean(confidences):.4f}")
    print(f"   - Min Confidence: {np.min(confidences):.4f}")
    print(f"   - Max Confidence: {np.max(confidences):.4f}")
    if y_test is not None:
        print(f"   - Accuracy: {accuracy:.2f}%")
        print(f"   - Up Predictions: {sum(predictions)}")
        print(f"   - Actual Up: {sum(y_test)}")
    
    plt.close()

def train_model(data_path, epochs=100, lr=0.1, batch_size=None, save_model=True, 
                model_dir=None, device='cpu', early_stopping_patience=30, 
                show_graph=False, save_metric='f1'):
    """앙상블 모델 학습"""
    print(f"\n{'='*60}")
    print(f"🚀 Ensemble Model Training (XGBoost + LightGBM)")
    print(f"{'='*60}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # 모델 저장 디렉토리 설정
    if model_dir is None:
        model_dir = Path("D:/stock/_v5/models")
    else:
        model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # 데이터 로드
    print("📦 Loading data...")
    X, y, input_dim = load_data(data_path)
    
    print(f"Input dimension: {input_dim}")
    print(f"Total samples: {len(X)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # K-Fold Cross-Validation 설정
    n_folds = 5
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    print(f"\n🔄 K-Fold Cross-Validation (n={n_folds})")
    print(f"  - Epochs (n_estimators): {epochs}")
    print(f"  - Learning rate: {lr}")
    print(f"  - Model save path: {model_dir}")
    print(f"  - Early Stopping Patience: {early_stopping_patience}")
    print(f"  - Save metric: {save_metric}")
    
    # 메트릭 초기화
    fold_metrics = {
        'acc': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    best_val_acc = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0
    best_val_f1 = 0.0
    best_model_path = None
    best_fold = 0
    best_model = None
    
    # 메트릭 이름 매핑
    metric_names = {
        'acc': 'Accuracy',
        'prec': 'Precision',
        'rec': 'Recall',
        'f1': 'F1 Score'
    }
    
    # 모델 파라미터
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'learning_rate': lr,
        'n_estimators': epochs,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    lgbm_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': lr,
        'n_estimators': epochs,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    # K-Fold 학습
    models = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
        print(f"\n{'='*60}")
        print(f"📊 Fold {fold_idx}/{n_folds}")
        print(f"{'='*60}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # 모델 초기화
        model = EnsembleModel(input_dim, xgb_params=xgb_params, lgbm_params=lgbm_params)
        
        # 학습
        print("\n🌳 Training ensemble model...")
        model.fit(X_train, y_train, X_val, y_val, 
                  xgb_early_stopping_rounds=early_stopping_patience if early_stopping_patience > 0 else None,
                  lgbm_early_stopping_rounds=early_stopping_patience if early_stopping_patience > 0 else None)
        
        # 검증 예측
        print("\n📊 Evaluating on validation set...")
        val_pred = model.predict(X_val)
        
        # 메트릭 계산
        val_acc = accuracy_score(y_val, val_pred) * 100
        precision = precision_score(y_val, val_pred, average='binary', zero_division=0)
        recall = recall_score(y_val, val_pred, average='binary', zero_division=0)
        f1 = f1_score(y_val, val_pred, average='binary', zero_division=0)
        
        # Fold 메트릭 저장
        fold_metrics['acc'].append(val_acc)
        fold_metrics['precision'].append(precision)
        fold_metrics['recall'].append(recall)
        fold_metrics['f1'].append(f1)
        
        print(f"Fold {fold_idx} Results:")
        print(f"  Acc: {val_acc:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        # 선택된 메트릭 값
        if save_metric == 'acc':
            current_metric = val_acc / 100.0
            best_metric = best_val_acc / 100.0
        elif save_metric == 'prec':
            current_metric = precision
            best_metric = best_val_precision
        elif save_metric == 'rec':
            current_metric = recall
            best_metric = best_val_recall
        elif save_metric == 'f1':
            current_metric = f1
            best_metric = best_val_f1
        else:
            current_metric = f1
            best_metric = best_val_f1
        
        # 최고 성능 모델 업데이트
        if current_metric > best_metric:
            best_val_acc = val_acc
            best_val_precision = precision
            best_val_recall = recall
            best_val_f1 = f1
            best_fold = fold_idx
            best_model = model
        
        models.append(model)
    
    # 평균 메트릭 계산
    mean_acc = np.mean(fold_metrics['acc'])
    mean_precision = np.mean(fold_metrics['precision'])
    mean_recall = np.mean(fold_metrics['recall'])
    mean_f1 = np.mean(fold_metrics['f1'])
    
    std_acc = np.std(fold_metrics['acc'])
    std_precision = np.std(fold_metrics['precision'])
    std_recall = np.std(fold_metrics['recall'])
    std_f1 = np.std(fold_metrics['f1'])
    
    print(f"\n{'='*60}")
    print(f"📊 K-Fold Cross-Validation Results (n={n_folds})")
    print(f"{'='*60}")
    print(f"Mean ± Std:")
    print(f"  Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Precision: {mean_precision:.4f} ± {std_precision:.4f}")
    print(f"  Recall: {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"  F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"\nBest Fold: {best_fold} ({metric_names.get(save_metric, save_metric.upper())} based)")
    print(f"  Accuracy: {best_val_acc:.2f}%")
    print(f"  Precision: {best_val_precision:.4f}")
    print(f"  Recall: {best_val_recall:.4f}")
    print(f"  F1 Score: {best_val_f1:.4f}")
    
    # 최고 성능 모델 사용
    model = best_model
    
    if save_model:
        # 파일명 생성
        if save_metric == 'acc':
            metric_str = f"acc_{best_val_acc:.2f}"
        elif save_metric == 'prec':
            metric_str = f"precision_{best_val_precision:.4f}"
        elif save_metric == 'rec':
            metric_str = f"recall_{best_val_recall:.4f}"
        elif save_metric == 'f1':
            metric_str = f"f1_{best_val_f1:.4f}"
        else:
            metric_str = f"f1_{best_val_f1:.4f}"
        
        best_model_path = model_dir / f"best_model_v2_kfold{n_folds}_fold{best_fold}_acc_{best_val_acc:.2f}_{metric_str}_f1_{best_val_f1:.4f}.pkl"
        
        # 모델 저장
        model_data = {
            'xgb_model': model.xgb_model,
            'lgbm_model': model.lgbm_model,
            'ensemble_weights': model.ensemble_weights,
            'input_dim': input_dim,
            'val_acc': best_val_acc,
            'val_precision': best_val_precision,
            'val_recall': best_val_recall,
            'val_f1': best_val_f1,
            'save_metric': save_metric,
            'kfold': n_folds,
            'best_fold': best_fold,
            'mean_metrics': {
                'acc': mean_acc,
                'precision': mean_precision,
                'recall': mean_recall,
                'f1': mean_f1
            },
            'std_metrics': {
                'acc': std_acc,
                'precision': std_precision,
                'recall': std_recall,
                'f1': std_f1
            }
        }
        joblib.dump(model_data, best_model_path)
        print(f"\n✨ Best model saved (Fold {best_fold}): {best_model_path}")
    
    if best_model_path:
        metric_display = metric_names.get(save_metric, save_metric.upper())
        print(f"\n✅ Best model saved ({metric_display} based): {best_model_path}")
        print(f"   Validation Accuracy: {best_val_acc:.2f}%")
        print(f"   Precision: {best_val_precision:.4f}{' ⭐' if save_metric == 'prec' else ''}")
        print(f"   Recall: {best_val_recall:.4f}{' ⭐' if save_metric == 'rec' else ''}")
        print(f"   F1 Score: {best_val_f1:.4f}{' ⭐' if save_metric == 'f1' else ''}")
        if save_metric == 'acc':
            print(f"   Accuracy: {best_val_acc:.2f}% ⭐")
        
        # 그래프 시각화
        if show_graph:
            # 최고 모델 로드
            model_data = joblib.load(best_model_path)
            model.xgb_model = model_data['xgb_model']
            model.lgbm_model = model_data['lgbm_model']
            model.ensemble_weights = model_data['ensemble_weights']
            visualize_predictions(model, data_path, model_dir=model_dir)
    else:
        print("\n⚠️ No model saved.")
        if show_graph:
            print("⚠️ Graph requires saved model.")
    
    return model, best_model_path

def main():
    parser = argparse.ArgumentParser(
        description="Stock Prediction Model Training (XGBoost + LightGBM Ensemble)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Basic training
  python train_v2.py --data preprocessed_005930_20240101_20241231.csv
  
  # With hyperparameters
  python train_v2.py --data preprocessed_005930_20240101_20241231.csv --epochs 200 --lr 0.05
  
  # Save metric selection
  python train_v2.py --data preprocessed_005930_20240101_20241231.csv --save-metric f1
  
  # With graph
  python train_v2.py --data preprocessed_005930_20240101_20241231.csv --graph
        """
    )
    
    parser.add_argument("--data", type=str, required=True, 
                       help="Preprocessed data file path")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="Number of estimators (default: 100)")
    parser.add_argument("--lr", type=float, default=0.1, 
                       help="Learning rate (default: 0.1)")
    parser.add_argument("--model-dir", type=str, default=None,
                       help="Model save directory (default: D:/stock/_v5/models)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save model")
    parser.add_argument("--early-stopping", type=int, default=30,
                       help="Early stopping patience (default: 30, 0 to disable)")
    parser.add_argument("--save-metric", type=str, default="f1", 
                       choices=["acc", "prec", "rec", "f1"],
                       help="Metric for model saving (default: f1)")
    parser.add_argument("--graph", action="store_true",
                       help="Generate prediction graph for last 30 samples after training")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device (default: cpu, not used for tree models)")
    
    args = parser.parse_args()
    
    # 데이터 파일 경로 처리
    data_path = args.data
    if not os.path.isabs(data_path):
        base_dir = Path("D:/stock/_v5/_data")
        data_path = base_dir / data_path
        data_path = str(data_path)
    
    try:
        train_model(
            data_path=data_path,
            epochs=args.epochs,
            lr=args.lr,
            save_model=not args.no_save,
            model_dir=args.model_dir,
            early_stopping_patience=args.early_stopping,
            show_graph=args.graph,
            save_metric=args.save_metric
        )
        print(f"\n{'='*60}")
        print(f"✨ Training completed!")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

