import torch
import torch.nn as nn
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

from s04_dataset import get_dataloaders
from s05_architecture import MultiScaleEnsemble
from focal_loss import FocalLoss

def visualize_predictions(model, data_path, device='cpu', num_samples=30, model_dir=None):
    """마지막 N개 데이터에 대한 예측 결과 시각화"""
    print(f"\n{'='*60}")
    print(f"📊 예측 결과 시각화 (마지막 {num_samples}개 데이터)")
    print(f"{'='*60}")
    
    # 데이터 로드
    df = pd.read_csv(data_path)
    
    # 날짜 컬럼 확인
    has_date = '날짜' in df.columns
    
    # Target과 Feature 분리
    if 'target' in df.columns:
        targets = df['target'].values
        features_df = df.drop(columns=['target'])
    else:
        targets = None
        features_df = df.copy()
    
    if has_date:
        dates = df['날짜'].copy()
        features_df = features_df.drop(columns=['날짜'])
    else:
        dates = None
    
    # 마지막 num_samples개 데이터 추출
    window_size = 60  # 모델의 윈도우 크기
    if len(features_df) < window_size + num_samples:
        num_samples = len(features_df) - window_size
        print(f"⚠️ 데이터가 부족하여 {num_samples}개만 시각화합니다.")
    
    model.eval()
    predictions = []
    confidences = []
    actuals = []
    date_list = []
    
    with torch.no_grad():
        for i in range(len(features_df) - window_size - num_samples, len(features_df) - window_size):
            # 윈도우 데이터 추출
            window_data = features_df.iloc[i:i+window_size].values
            data_tensor = torch.FloatTensor(window_data).unsqueeze(0).to(device)
            
            # 예측
            output = model(data_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0][pred_class].item()
            
            predictions.append(pred_class)
            confidences.append(confidence)
            
            if targets is not None:
                actuals.append(targets[i + window_size])
            
            if dates is not None:
                date_list.append(str(dates.iloc[i + window_size]))
            else:
                date_list.append(f"Day {i + window_size}")
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. Prediction vs Actual
    x = range(len(predictions))
    axes[0].plot(x, predictions, 'o-', label='Prediction', color='blue', linewidth=2, markersize=6)
    if actuals:
        axes[0].plot(x, actuals, 's-', label='Actual', color='red', linewidth=2, markersize=6)
        # Accuracy calculation
        accuracy = (np.array(predictions) == np.array(actuals)).mean() * 100
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
    
    # X축 레이블 (날짜가 있으면 날짜 표시)
    if dates is not None and len(date_list) <= 30:
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([d.split()[0] if ' ' in d else d for d in date_list], rotation=45, ha='right')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([d.split()[0] if ' ' in d else d for d in date_list], rotation=45, ha='right')
    
    plt.tight_layout()
    
    # 그래프 저장
    if model_dir is None:
        model_dir = Path("D:/stock/_v5/models")
    else:
        model_dir = Path(model_dir)
    
    graph_path = model_dir / f"prediction_graph_{num_samples}samples.png"
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    print(f"✅ 그래프 저장: {graph_path}")
    
    # Statistics output
    print(f"\n📊 Prediction Statistics:")
    print(f"   - Mean Confidence: {np.mean(confidences):.4f}")
    print(f"   - Min Confidence: {np.min(confidences):.4f}")
    print(f"   - Max Confidence: {np.max(confidences):.4f}")
    if actuals:
        print(f"   - Accuracy: {accuracy:.2f}%")
        print(f"   - Up Predictions: {sum(predictions)}")
        print(f"   - Actual Up: {sum(actuals)}")
    
    plt.close()

def train_model(data_path, epochs=50, lr=0.001, batch_size=32, save_model=True, model_dir=None, device='cpu', early_stopping_patience=30, use_focal_loss=False, focal_alpha=1.0, focal_gamma=2.0, show_graph=False, save_metric='prec'):
    """모델 학습"""
    print(f"\n{'='*60}")
    print(f"🚀 모델 학습 시작")
    print(f"{'='*60}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터 파일이 없습니다: {data_path}")
    
    # 디바이스 설정
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
        device = "cpu"
    
    device = torch.device(device)
    print(f"사용 디바이스: {device}")
    
    # 모델 저장 디렉토리 설정
    if model_dir is None:
        model_dir = Path("D:/stock/_v5/models")
    else:
        model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # 데이터로더 생성
    print("📦 데이터로더 생성 중...")
    train_loader, val_loader = get_dataloaders(data_path, batch_size=batch_size)
    
    # 데이터셋의 피처 수 확인
    sample_x, _ = next(iter(train_loader))
    input_dim = sample_x.shape[2]
    
    print(f"입력 차원: {input_dim}")
    print(f"학습 배치 수: {len(train_loader)}, 검증 배치 수: {len(val_loader)}")
    
    # 모델 초기화 및 디바이스로 이동
    model = MultiScaleEnsemble(input_dim)
    model = model.to(device)
    
    # 클래스 가중치 계산 (불균형 데이터 처리)
    # 전체 데이터셋의 클래스 분포 확인
    print("📊 클래스 분포 확인 중...")
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)
    class_counts_full = np.bincount(all_labels)
    
    # 손실 함수 선택
    if use_focal_loss:
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        print(f"📊 손실 함수: Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
    else:
        if len(class_counts_full) == 2:
            # 클래스 가중치 계산 (적은 클래스에 더 높은 가중치)
            total = class_counts_full.sum()
            class_weights = torch.tensor([
                total / (len(class_counts_full) * class_counts_full[0]),
                total / (len(class_counts_full) * class_counts_full[1])
            ], dtype=torch.float32).to(device)
            print(f"📊 클래스 분포: {class_counts_full}")
            print(f"📊 클래스 가중치: {class_weights.cpu().numpy()}")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
            print(f"📊 손실 함수: CrossEntropyLoss")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning Rate Scheduler 추가
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6
    )
    
    # Early Stopping 설정
    early_stopping_counter = 0
    
    print(f"\n학습 시작...")
    print(f"  - 에포크: {epochs}")
    print(f"  - 학습률: {lr}")
    print(f"  - 배치 크기: {batch_size}")
    print(f"  - 디바이스: {device}")
    print(f"  - 모델 저장 경로: {model_dir}")
    print(f"  - Early Stopping Patience: {early_stopping_patience}")
    print(f"  - Learning Rate Scheduler: ReduceLROnPlateau")
    
    # 저장 기준 메트릭 초기화
    best_val_acc = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0
    best_val_f1 = 0.0
    best_model_path = None
    
    # 메트릭 이름 매핑
    metric_names = {
        'acc': '정확도',
        'prec': 'Precision',
        'rec': 'Recall',
        'f1': 'F1 Score'
    }
    
    print(f"  - 모델 저장 기준: {metric_names.get(save_metric, save_metric)}")
    
    for epoch in range(epochs):
        # 학습
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            # 데이터를 디바이스로 이동
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            # Gradient Clipping 추가
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # 검증
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for val_x, val_y in val_loader:
                # 데이터를 디바이스로 이동
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                
                val_output = model(val_x)
                pred = val_output.argmax(dim=1)
                
                # CPU로 이동하여 리스트에 추가
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(val_y.cpu().numpy())
        
        # 메트릭 계산
        train_loss_avg = train_loss / len(train_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        val_acc = (all_preds == all_labels).mean() * 100
        
        # Precision, Recall, F1 Score 계산
        precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        
        # 선택된 메트릭에 따라 Learning Rate Scheduler 업데이트
        if save_metric == 'acc':
            current_metric = val_acc / 100.0  # 백분율을 소수로 변환
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
            current_metric = precision
            best_metric = best_val_precision
        
        scheduler.step(current_metric)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 최고 성능 모델 저장 (선택된 메트릭 기준)
        improved = False
        if current_metric > best_metric:
            # 모든 메트릭 업데이트
            best_val_acc = val_acc
            best_val_precision = precision
            best_val_recall = recall
            best_val_f1 = f1
            
            improved = True
            early_stopping_counter = 0
            if save_model:
                # 기존 모델 삭제 (선택사항)
                if best_model_path and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                
                # 파일명에 선택된 메트릭 강조
                metric_value = current_metric
                if save_metric == 'acc':
                    metric_value = val_acc
                    metric_str = f"acc_{val_acc:.2f}"
                elif save_metric == 'prec':
                    metric_str = f"precision_{precision:.4f}"
                elif save_metric == 'rec':
                    metric_str = f"recall_{recall:.4f}"
                elif save_metric == 'f1':
                    metric_str = f"f1_{f1:.4f}"
                else:
                    metric_str = f"precision_{precision:.4f}"
                
                best_model_path = model_dir / f"best_model_epoch_{epoch+1}_acc_{val_acc:.2f}_{metric_str}_f1_{f1:.4f}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_precision': precision,
                    'val_recall': recall,
                    'val_f1': f1,
                    'input_dim': input_dim,
                    'save_metric': save_metric
                }, best_model_path)
        else:
            early_stopping_counter += 1
        
        # Early Stopping 체크
        if early_stopping_patience > 0 and early_stopping_counter >= early_stopping_patience:
            print(f"\n⚠️ Early Stopping: {early_stopping_patience} 에포크 동안 개선이 없어 학습을 중단합니다.")
            break
        
        # Best 메트릭 표시
        if save_metric == 'acc':
            best_metric_str = f"Best Acc: {best_val_acc:.2f}%"
        elif save_metric == 'prec':
            best_metric_str = f"Best Precision: {best_val_precision:.4f}"
        elif save_metric == 'rec':
            best_metric_str = f"Best Recall: {best_val_recall:.4f}"
        elif save_metric == 'f1':
            best_metric_str = f"Best F1: {best_val_f1:.4f}"
        else:
            best_metric_str = f"Best Precision: {best_val_precision:.4f}"
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_loss_avg:.4f} | "
              f"Val Acc: {val_acc:.2f}% | Precision: {precision:.4f} | "
              f"Recall: {recall:.4f} | F1: {f1:.4f} | {best_metric_str} | "
              f"LR: {current_lr:.2e} | {'✨' if improved else ''}")
    
    if best_model_path:
        # 최종 모델의 메트릭 로드
        checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
        best_precision = checkpoint.get('val_precision', 0)
        best_recall = checkpoint.get('val_recall', 0)
        best_f1 = checkpoint.get('val_f1', 0)
        
        metric_display = metric_names.get(save_metric, save_metric.upper())
        print(f"\n✅ 최고 성능 모델 저장 ({metric_display} 기준): {best_model_path}")
        print(f"   검증 정확도: {best_val_acc:.2f}%")
        print(f"   Precision: {best_precision:.4f}{' ⭐' if save_metric == 'prec' else ''}")
        print(f"   Recall: {best_recall:.4f}{' ⭐' if save_metric == 'rec' else ''}")
        print(f"   F1 Score: {best_f1:.4f}{' ⭐' if save_metric == 'f1' else ''}")
        if save_metric == 'acc':
            print(f"   정확도: {best_val_acc:.2f}% ⭐")
        
        # 그래프 시각화
        if show_graph:
            # 최고 모델 로드
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            visualize_predictions(model, data_path, device=device, model_dir=model_dir)
    else:
        print("\n⚠️ 저장된 모델이 없습니다.")
        if show_graph:
            print("⚠️ 그래프를 생성하려면 모델이 저장되어야 합니다.")
    
    return model, best_model_path

def main():
    parser = argparse.ArgumentParser(
        description="주식 예측 모델 학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 학습
  python train.py --data preprocessed_005930_20240101_20241231.csv
  
  # 하이퍼파라미터 조정
  python train.py --data preprocessed_005930_20240101_20241231.csv --epochs 100 --lr 0.0001 --batch-size 64
  
  # 모델 저장 경로 지정
  python train.py --data preprocessed_005930_20240101_20241231.csv --model-dir ./my_models
  
  # GPU 사용
  python train.py --data preprocessed_005930_20240101_20241231.csv --device cuda
  
  # Focal Loss 사용 (불균형 데이터에 효과적)
  python train.py --data preprocessed_005930_20240101_20241231.csv --focal-loss --focal-gamma 2.0
  
  # Focal Loss + 커스텀 파라미터
  python train.py --data preprocessed_005930_20240101_20241231.csv --focal-loss --focal-alpha 0.25 --focal-gamma 2.0
        """
    )
    
    parser.add_argument("--data", type=str, required=True, 
                       help="전처리된 데이터 파일 경로")
    parser.add_argument("--epochs", type=int, default=300, 
                       help="학습 에포크 수 (기본값: 50)")
    parser.add_argument("--lr", type=float, default=0.001, 
                       help="학습률 (기본값: 0.001)")
    parser.add_argument("--batch-size", type=int, default=32, 
                       help="배치 크기 (기본값: 32)")
    parser.add_argument("--model-dir", type=str, default=None,
                       help="모델 저장 디렉토리 (기본값: D:/stock/_v5/models)")
    parser.add_argument("--no-save", action="store_true",
                       help="모델 저장하지 않기")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"],
                       help="사용할 디바이스 (기본값: cpu)")
    parser.add_argument("--early-stopping", type=int, default=30,
                       help="Early stopping patience (기본값: 30, 0이면 비활성화)")
    parser.add_argument("--focal-loss", action="store_true",
                       help="Focal Loss 사용 (불균형 데이터에 효과적)")
    parser.add_argument("--focal-alpha", type=float, default=1.0,
                       help="Focal Loss alpha 파라미터 (기본값: 1.0)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                       help="Focal Loss gamma 파라미터 (기본값: 2.0)")
    parser.add_argument("--graph", action="store_true",
                       help="학습 완료 후 마지막 30개 데이터에 대한 예측 결과 그래프 생성")
    parser.add_argument("--save-metric", type=str, default="f1", 
                       choices=["acc", "prec", "rec", "f1"],
                       help="모델 저장 기준 메트릭 (기본값: prec)")
    
    args = parser.parse_args()
    
    # 데이터 파일 경로 처리 (상대 경로인 경우 _data 폴더 기준)
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
            batch_size=args.batch_size,
            save_model=not args.no_save,
            model_dir=args.model_dir,
            device=args.device,
            early_stopping_patience=args.early_stopping,
            use_focal_loss=args.focal_loss,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            show_graph=args.graph,
            save_metric=args.save_metric
        )
        print(f"\n{'='*60}")
        print(f"✨ 학습 완료!")
        print(f"{'='*60}")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

