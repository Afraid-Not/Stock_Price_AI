import pandas as pd
import numpy as np
import argparse
import os
import json
import joblib
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# 트리 기반 모델
import catboost as cb

# 딥러닝 모델
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

class TimeSeriesDataset(Dataset):
    """시계열 데이터셋"""
    def __init__(self, X, y, sequence_length=30):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        seq_X = self.X[idx:idx+self.sequence_length]
        target = self.y[idx+self.sequence_length-1]
        # float32로 변환하여 PyTorch와 호환성 확보
        seq_X = seq_X.astype(np.float32)
        # CrossEntropyLoss는 스칼라 텐서를 기대하므로 .item() 또는 직접 변환
        return torch.from_numpy(seq_X).float(), torch.tensor(int(target), dtype=torch.long)

class LSTMClassifier(nn.Module):
    """LSTM 분류 모델"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # 마지막 시퀀스만 사용
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        out = self.fc(out)
        return out

class EnsembleTrainer:
    def __init__(self, data_path, model_dir="D:/stock/_v8/models", n_splits=5, 
                 sequence_length=30, device='cuda' if torch.cuda.is_available() else 'cpu',
                 catboost_weight=0.5, lstm_weight=0.5):
        self.data_path = data_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.n_splits = n_splits
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.sequence_length = sequence_length
        self.device = device
        self.catboost_weight = catboost_weight
        self.lstm_weight = lstm_weight
        
        self.results = {}
        
    def load_data(self, test_size=50):
        """데이터 로드 및 분리"""
        print(f"📂 데이터 로드: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        if 'target' not in df.columns:
            raise ValueError("'target' 컬럼이 없습니다.")
        
        # 날짜 컬럼 제거
        if '날짜' in df.columns:
            df = df.drop(columns=['날짜'])
        
        # next_rtn이 있으면 제거 (타겟 생성에만 사용)
        if 'next_rtn' in df.columns:
            df = df.drop(columns=['next_rtn'])
        
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
        print(f"   Target 분포: {np.bincount(y_train_val)}")
        
        return X_train_val, y_train_val, X_test, y_test
    
    def train_catboost(self, X_train, y_train, X_val, y_val):
        """CatBoost 학습"""
        print("  🐱 CatBoost 학습 중...")
        # 클래스 불균형 처리
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        class_weights = [neg_count / pos_count if pos_count > 0 else 1.0, 1.0]
        
        model = cb.CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            eval_metric='Logloss',
            class_weights=class_weights,
            min_data_in_leaf=20,
            random_seed=42,
            verbose=False
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
        return model
    
    def train_lstm(self, X_train, y_train, X_val, y_val, input_size):
        """LSTM 학습"""
        print("  🔄 LSTM 학습 중...")
        
        # 시퀀스 데이터 생성
        train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, self.sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        model = LSTMClassifier(input_size=input_size, hidden_size=64, num_layers=2, num_classes=2).to(self.device)
        
        # 클래스 불균형 처리: Weighted Loss
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        if pos_count > 0 and neg_count > 0:
            weight = torch.tensor([neg_count / pos_count, 1.0], dtype=torch.float32).to(self.device)
        else:
            weight = torch.tensor([1.0, 1.0], dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = Adam(model.parameters(), lr=0.001)
        
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            # Train
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device).float()
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device).float()
                    batch_y = batch_y.to(self.device)
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        model.load_state_dict(best_model_state)
        return model
    
    def predict_catboost_proba(self, model, X):
        """CatBoost 예측 확률"""
        return model.predict_proba(X)
    
    def predict_lstm_proba(self, model, X, y):
        """LSTM 예측 확률"""
        model.eval()
        # 전체 데이터를 시퀀스로 변환 (float32로 변환)
        if len(X) < self.sequence_length:
            padding = np.tile(X[0:1], (self.sequence_length - len(X), 1))
            X_padded = np.vstack([padding, X]).astype(np.float32)
            y_padded = np.concatenate([[y[0]] * (self.sequence_length - len(y)), y])
        else:
            X_padded = X.astype(np.float32)
            y_padded = y
        
        dataset = TimeSeriesDataset(X_padded, y_padded, self.sequence_length)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        probabilities = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device).float()
                outputs = model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                probabilities.append(probs.cpu().numpy())
        
        prob_array = np.vstack(probabilities)
        
        # 시퀀스 길이만큼 앞부분은 예측 불가하므로 첫 번째 확률로 패딩
        if len(prob_array) > 0:
            padding = np.tile(prob_array[0:1], (self.sequence_length - 1, 1))
            return np.vstack([padding, prob_array])
        else:
            return np.array([])
    
    def ensemble_predict(self, catboost_model, lstm_model, X_test, y_test):
        """앙상블 예측 (가중 평균)"""
        # CatBoost 예측 확률
        cb_proba = self.predict_catboost_proba(catboost_model, X_test)
        
        # LSTM 예측 확률
        lstm_proba = self.predict_lstm_proba(lstm_model, X_test, y_test)
        
        # 시퀀스 길이만큼 앞부분 제거
        if len(lstm_proba) > len(cb_proba):
            lstm_proba_trimmed = lstm_proba[self.sequence_length-1:]
        else:
            lstm_proba_trimmed = lstm_proba
        
        # 길이 맞추기
        min_len = min(len(cb_proba), len(lstm_proba_trimmed))
        cb_proba = cb_proba[:min_len]
        lstm_proba_trimmed = lstm_proba_trimmed[:min_len]
        
        # 가중 평균
        ensemble_proba = (self.catboost_weight * cb_proba + 
                         self.lstm_weight * lstm_proba_trimmed)
        
        # 클래스 예측
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        
        return ensemble_pred, ensemble_proba
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """모델 평가"""
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        
        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = 0.0
        
        return {
            'accuracy': acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
    
    def train_and_evaluate(self, X_train_val, y_train_val, X_test, y_test):
        """앙상블 모델 학습 및 평가"""
        print(f"\n{'='*60}")
        print(f"🚀 CatBoost + LSTM 앙상블 학습 시작")
        print(f"   CatBoost 가중치: {self.catboost_weight}")
        print(f"   LSTM 가중치: {self.lstm_weight}")
        print(f"{'='*60}\n")
        
        input_size = X_train_val.shape[1]
        all_results = {}
        
        # KFold로 학습
        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(X_train_val)):
            print(f"\n📊 Fold {fold_idx + 1}/{self.n_splits}")
            print("-" * 60)
            
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
            
            # 1. CatBoost 학습
            catboost_model = self.train_catboost(X_train, y_train, X_val, y_val)
            
            # 2. LSTM 학습
            lstm_model = self.train_lstm(X_train, y_train, X_val, y_val, input_size)
            
            # 3. 개별 모델 평가
            cb_pred = catboost_model.predict(X_test)
            cb_metrics = self.evaluate_model(y_test, cb_pred, 'CatBoost')
            print(f"  ✅ CatBoost - Acc: {cb_metrics['accuracy']:.4f}, F1: {cb_metrics['f1_score']:.4f}")
            
            lstm_pred, _ = self.ensemble_predict(catboost_model, lstm_model, X_test, y_test)
            # 시퀀스 길이만큼 앞부분 제거
            y_test_trimmed = y_test[self.sequence_length-1:]
            lstm_pred_trimmed = lstm_pred[self.sequence_length-1:]
            min_len = min(len(y_test_trimmed), len(lstm_pred_trimmed))
            lstm_metrics = self.evaluate_model(y_test_trimmed[:min_len], lstm_pred_trimmed[:min_len], 'LSTM')
            print(f"  ✅ LSTM - Acc: {lstm_metrics['accuracy']:.4f}, F1: {lstm_metrics['f1_score']:.4f}")
            
            # 4. 앙상블 예측
            ensemble_pred, ensemble_proba = self.ensemble_predict(catboost_model, lstm_model, X_test, y_test)
            
            # 길이 맞추기
            min_len = min(len(y_test), len(ensemble_pred))
            y_test_final = y_test[:min_len]
            ensemble_pred_final = ensemble_pred[:min_len]
            
            ensemble_metrics = self.evaluate_model(y_test_final, ensemble_pred_final, 'Ensemble')
            print(f"  ✅ Ensemble - Acc: {ensemble_metrics['accuracy']:.4f}, F1: {ensemble_metrics['f1_score']:.4f}")
            
            all_results[f'fold_{fold_idx+1}'] = {
                'CatBoost': cb_metrics,
                'LSTM': lstm_metrics,
                'Ensemble': ensemble_metrics
            }
        
        # 결과 집계
        self.aggregate_results(all_results)
        
        return all_results
    
    def aggregate_results(self, all_results):
        """결과 집계 및 저장"""
        print(f"\n{'='*60}")
        print(f"📊 결과 집계")
        print(f"{'='*60}\n")
        
        model_names = ['CatBoost', 'LSTM', 'Ensemble']
        metrics_list = ['accuracy', 'f1_score', 'precision', 'recall', 'auc']
        
        summary = {}
        
        for model_name in model_names:
            model_metrics = {metric: [] for metric in metrics_list}
            
            for fold_key, fold_results in all_results.items():
                if model_name in fold_results:
                    for metric in metrics_list:
                        if metric in fold_results[model_name]:
                            model_metrics[metric].append(fold_results[model_name][metric])
            
            if any(len(v) > 0 for v in model_metrics.values()):
                summary[model_name] = {
                    metric: {
                        'mean': np.mean(model_metrics[metric]),
                        'std': np.std(model_metrics[metric])
                    }
                    for metric in metrics_list if len(model_metrics[metric]) > 0
                }
        
        # 결과 출력
        print(f"{'모델':<15} {'Accuracy':<15} {'F1 Score':<15} {'Precision':<15} {'Recall':<15} {'AUC':<15}")
        print("-" * 90)
        
        for model_name, metrics in summary.items():
            acc = metrics.get('accuracy', {}).get('mean', 0)
            f1 = metrics.get('f1_score', {}).get('mean', 0)
            prec = metrics.get('precision', {}).get('mean', 0)
            rec = metrics.get('recall', {}).get('mean', 0)
            auc = metrics.get('auc', {}).get('mean', 0)
            
            print(f"{model_name:<15} {acc:.4f}±{metrics.get('accuracy', {}).get('std', 0):.4f}  "
                  f"{f1:.4f}±{metrics.get('f1_score', {}).get('std', 0):.4f}  "
                  f"{prec:.4f}±{metrics.get('precision', {}).get('std', 0):.4f}  "
                  f"{rec:.4f}±{metrics.get('recall', {}).get('std', 0):.4f}  "
                  f"{auc:.4f}±{metrics.get('auc', {}).get('std', 0):.4f}")
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.model_dir / f"ensemble_results_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'weights': {
                    'catboost': self.catboost_weight,
                    'lstm': self.lstm_weight
                },
                'summary': summary,
                'detailed_results': all_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 결과 저장: {results_file}")
        
        # 시각화
        self.plot_comparison(summary, timestamp)
        
        self.results = summary
    
    def plot_comparison(self, summary, timestamp):
        """결과 비교 시각화"""
        model_names = list(summary.keys())
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            means = [summary[model][metric]['mean'] for model in model_names]
            stds = [summary[model][metric]['std'] for model in model_names]
            
            x_pos = np.arange(len(model_names))
            ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            ax.set_xlabel('Model')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.model_dir / f"ensemble_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 시각화 저장: {plot_file}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="CatBoost + LSTM 앙상블 학습")
    parser.add_argument("--data", type=str, required=True, help="전처리된 데이터 파일 경로")
    parser.add_argument("--model-dir", type=str, default="D:/stock/_v8/models", help="모델 저장 디렉토리")
    parser.add_argument("--n-splits", type=int, default=5, help="KFold 분할 수")
    parser.add_argument("--sequence-length", type=int, default=30, help="시계열 시퀀스 길이")
    parser.add_argument("--test-size", type=int, default=50, help="테스트 세트 크기")
    parser.add_argument("--catboost-weight", type=float, default=0.5, help="CatBoost 가중치")
    parser.add_argument("--lstm-weight", type=float, default=0.5, help="LSTM 가중치")
    
    args = parser.parse_args()
    
    # 가중치 합이 1이 되도록 정규화
    total_weight = args.catboost_weight + args.lstm_weight
    if total_weight > 0:
        args.catboost_weight /= total_weight
        args.lstm_weight /= total_weight
    
    trainer = EnsembleTrainer(
        data_path=args.data,
        model_dir=args.model_dir,
        n_splits=args.n_splits,
        sequence_length=args.sequence_length,
        catboost_weight=args.catboost_weight,
        lstm_weight=args.lstm_weight
    )
    
    X_train_val, y_train_val, X_test, y_test = trainer.load_data(test_size=args.test_size)
    results = trainer.train_and_evaluate(X_train_val, y_train_val, X_test, y_test)
    
    print(f"\n{'='*60}")
    print(f"✅ 앙상블 모델 학습 및 비교 완료!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()




