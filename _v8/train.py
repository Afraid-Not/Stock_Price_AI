import pandas as pd
import numpy as np
import argparse
import os
import json
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# 트리 기반 모델
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# 딥러닝 모델
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# 시계열 모델을 위한 추가 라이브러리
try:
    from transformers import InformerConfig, InformerModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ transformers 라이브러리가 없습니다. Informer는 건너뜁니다.")

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

class TransformerClassifier(nn.Module):
    """Transformer 분류 모델"""
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, num_classes=2, dropout=0.2):
        super(TransformerClassifier, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.transformer(x)
        # 마지막 시퀀스만 사용
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x

class TFTModel(nn.Module):
    """Temporal Fusion Transformer (간단한 버전)"""
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, num_classes=2, dropout=0.2):
        super(TFTModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # TFT는 attention 메커니즘을 사용
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        # Self-attention
        x, _ = self.attention(x, x, x)
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ModelComparisonTrainer:
    def __init__(self, data_path, model_dir="D:/stock/_v8/models", n_splits=5, 
                 sequence_length=30, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.data_path = data_path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.n_splits = n_splits
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        self.sequence_length = sequence_length
        self.device = device
        
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
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """XGBoost 학습"""
        print("  🌳 XGBoost 학습 중...")
        # 클래스 불균형 처리
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight('balanced', y_train)
        
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]) if len(y_train[y_train==1]) > 0 else 1.0
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                 sample_weight=sample_weight, verbose=False)
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """LightGBM 학습"""
        print("  🌲 LightGBM 학습 중...")
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 클래스 불균형 처리
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_split_gain': 0.1,
            'scale_pos_weight': scale_pos_weight,
            'verbose': -1,
            'seed': 42
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        return model
    
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
    
    def train_transformer(self, X_train, y_train, X_val, y_val, input_size):
        """Transformer 학습"""
        print("  ⚡ Transformer 학습 중...")
        
        train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, self.sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        model = TransformerClassifier(input_size=input_size, d_model=64, nhead=4, 
                                     num_layers=2, num_classes=2).to(self.device)
        
        # 클래스 불균형 처리
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        if pos_count > 0 and neg_count > 0:
            weight = torch.tensor([neg_count / pos_count, 1.0]).to(self.device)
        else:
            weight = torch.tensor([1.0, 1.0]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = Adam(model.parameters(), lr=0.001)
        
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device).float()
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
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
    
    def train_tft(self, X_train, y_train, X_val, y_val, input_size):
        """TFT 학습"""
        print("  🎯 TFT 학습 중...")
        
        train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, self.sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        model = TFTModel(input_size=input_size, d_model=64, nhead=4, 
                        num_layers=2, num_classes=2).to(self.device)
        
        # 클래스 불균형 처리
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        if pos_count > 0 and neg_count > 0:
            weight = torch.tensor([neg_count / pos_count, 1.0]).to(self.device)
        else:
            weight = torch.tensor([1.0, 1.0]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = Adam(model.parameters(), lr=0.001)
        
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(50):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device).float()
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
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
    
    def train_informer(self, X_train, y_train, X_val, y_val, input_size):
        """Informer 학습 (간단한 구현)"""
        if not HAS_TRANSFORMERS:
            return None
        
        print("  📊 Informer 학습 중...")
        # Informer는 복잡하므로 간단한 Transformer 기반 구현 사용
        return self.train_transformer(X_train, y_train, X_val, y_val, input_size)
    
    def predict_xgboost(self, model, X):
        """XGBoost 예측"""
        return model.predict(X)
    
    def predict_lightgbm(self, model, X):
        """LightGBM 예측"""
        return model.predict(X).astype(int)
    
    def predict_catboost(self, model, X):
        """CatBoost 예측"""
        return model.predict(X)
    
    def predict_deep(self, model, X, y):
        """딥러닝 모델 예측"""
        model.eval()
        # 전체 데이터를 시퀀스로 변환 (float32로 변환)
        if len(X) < self.sequence_length:
            # 데이터가 시퀀스 길이보다 짧으면 첫 번째 값으로 패딩
            padding = np.tile(X[0:1], (self.sequence_length - len(X), 1))
            X_padded = np.vstack([padding, X]).astype(np.float32)
            y_padded = np.concatenate([[y[0]] * (self.sequence_length - len(y)), y])
        else:
            X_padded = X.astype(np.float32)
            y_padded = y
        
        dataset = TimeSeriesDataset(X_padded, y_padded, self.sequence_length)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_X, _ in loader:
                batch_X = batch_X.to(self.device).float()
                outputs = model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        # 시퀀스 길이만큼 앞부분은 예측 불가하므로 첫 번째 예측값으로 패딩
        if len(predictions) > 0:
            padding = np.array([predictions[0]] * (self.sequence_length - 1))
            return np.concatenate([padding, predictions])
        else:
            return np.array([])
    
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
    
    def train_and_evaluate_all(self, X_train_val, y_train_val, X_test, y_test):
        """모든 모델 학습 및 평가"""
        print(f"\n{'='*60}")
        print(f"🚀 모든 모델 학습 및 비교 시작")
        print(f"{'='*60}\n")
        
        input_size = X_train_val.shape[1]
        all_results = {}
        
        # KFold로 학습
        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(X_train_val)):
            print(f"\n📊 Fold {fold_idx + 1}/{self.n_splits}")
            print("-" * 60)
            
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
            
            fold_results = {}
            
            # 1. XGBoost
            try:
                model = self.train_xgboost(X_train, y_train, X_val, y_val)
                y_pred_test = self.predict_xgboost(model, X_test)
                metrics = self.evaluate_model(y_test, y_pred_test, 'XGBoost')
                fold_results['XGBoost'] = metrics
                print(f"  ✅ XGBoost - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
            except Exception as e:
                print(f"  ❌ XGBoost 실패: {e}")
                fold_results['XGBoost'] = None
            
            # 2. LightGBM
            try:
                model = self.train_lightgbm(X_train, y_train, X_val, y_val)
                y_pred_test = self.predict_lightgbm(model, X_test)
                metrics = self.evaluate_model(y_test, y_pred_test, 'LightGBM')
                fold_results['LightGBM'] = metrics
                print(f"  ✅ LightGBM - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
            except Exception as e:
                print(f"  ❌ LightGBM 실패: {e}")
                fold_results['LightGBM'] = None
            
            # 3. CatBoost
            try:
                model = self.train_catboost(X_train, y_train, X_val, y_val)
                y_pred_test = self.predict_catboost(model, X_test)
                metrics = self.evaluate_model(y_test, y_pred_test, 'CatBoost')
                fold_results['CatBoost'] = metrics
                print(f"  ✅ CatBoost - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
            except Exception as e:
                print(f"  ❌ CatBoost 실패: {e}")
                fold_results['CatBoost'] = None
            
            # 4. LSTM
            try:
                model = self.train_lstm(X_train, y_train, X_val, y_val, input_size)
                y_pred_test = self.predict_deep(model, X_test, y_test)
                # 시퀀스 길이만큼 앞부분 제거
                y_test_trimmed = y_test[self.sequence_length-1:]
                y_pred_test_trimmed = y_pred_test[self.sequence_length-1:]
                metrics = self.evaluate_model(y_test_trimmed, y_pred_test_trimmed, 'LSTM')
                fold_results['LSTM'] = metrics
                print(f"  ✅ LSTM - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
            except Exception as e:
                print(f"  ❌ LSTM 실패: {e}")
                fold_results['LSTM'] = None
            
            # 5. Transformer
            try:
                model = self.train_transformer(X_train, y_train, X_val, y_val, input_size)
                y_pred_test = self.predict_deep(model, X_test, y_test)
                y_test_trimmed = y_test[self.sequence_length-1:]
                y_pred_test_trimmed = y_pred_test[self.sequence_length-1:]
                metrics = self.evaluate_model(y_test_trimmed, y_pred_test_trimmed, 'Transformer')
                fold_results['Transformer'] = metrics
                print(f"  ✅ Transformer - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
            except Exception as e:
                print(f"  ❌ Transformer 실패: {e}")
                fold_results['Transformer'] = None
            
            # 6. TFT
            try:
                model = self.train_tft(X_train, y_train, X_val, y_val, input_size)
                y_pred_test = self.predict_deep(model, X_test, y_test)
                y_test_trimmed = y_test[self.sequence_length-1:]
                y_pred_test_trimmed = y_pred_test[self.sequence_length-1:]
                metrics = self.evaluate_model(y_test_trimmed, y_pred_test_trimmed, 'TFT')
                fold_results['TFT'] = metrics
                print(f"  ✅ TFT - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
            except Exception as e:
                print(f"  ❌ TFT 실패: {e}")
                fold_results['TFT'] = None
            
            # 7. Informer
            try:
                model = self.train_informer(X_train, y_train, X_val, y_val, input_size)
                if model is not None:
                    y_pred_test = self.predict_deep(model, X_test, y_test)
                    y_test_trimmed = y_test[self.sequence_length-1:]
                    y_pred_test_trimmed = y_pred_test[self.sequence_length-1:]
                    metrics = self.evaluate_model(y_test_trimmed, y_pred_test_trimmed, 'Informer')
                    fold_results['Informer'] = metrics
                    print(f"  ✅ Informer - Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
                else:
                    fold_results['Informer'] = None
            except Exception as e:
                print(f"  ❌ Informer 실패: {e}")
                fold_results['Informer'] = None
            
            all_results[f'fold_{fold_idx+1}'] = fold_results
        
        # 결과 집계
        self.aggregate_results(all_results)
        
        return all_results
    
    def aggregate_results(self, all_results):
        """결과 집계 및 저장"""
        print(f"\n{'='*60}")
        print(f"📊 결과 집계")
        print(f"{'='*60}\n")
        
        model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'LSTM', 'Transformer', 'TFT', 'Informer']
        metrics_list = ['accuracy', 'f1_score', 'precision', 'recall', 'auc']
        
        summary = {}
        
        for model_name in model_names:
            model_metrics = {metric: [] for metric in metrics_list}
            
            for fold_key, fold_results in all_results.items():
                if model_name in fold_results and fold_results[model_name] is not None:
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
        results_file = self.model_dir / f"model_comparison_{timestamp}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
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
        plot_file = self.model_dir / f"model_comparison_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 시각화 저장: {plot_file}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="여러 모델 비교 학습")
    parser.add_argument("--data", type=str, required=True, help="전처리된 데이터 파일 경로")
    parser.add_argument("--model-dir", type=str, default="D:/stock/_v8/models", help="모델 저장 디렉토리")
    parser.add_argument("--n-splits", type=int, default=5, help="KFold 분할 수")
    parser.add_argument("--sequence-length", type=int, default=30, help="시계열 시퀀스 길이")
    parser.add_argument("--test-size", type=int, default=50, help="테스트 세트 크기")
    
    args = parser.parse_args()
    
    trainer = ModelComparisonTrainer(
        data_path=args.data,
        model_dir=args.model_dir,
        n_splits=args.n_splits,
        sequence_length=args.sequence_length
    )
    
    X_train_val, y_train_val, X_test, y_test = trainer.load_data(test_size=args.test_size)
    results = trainer.train_and_evaluate_all(X_train_val, y_train_val, X_test, y_test)
    
    print(f"\n{'='*60}")
    print(f"✅ 모든 모델 학습 및 비교 완료!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

