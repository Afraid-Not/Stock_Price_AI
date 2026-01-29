import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

from s05_architecture import MultiScaleEnsemble

def load_model(model_path, device='cpu'):
    """저장된 모델 가중치 로드"""
    print(f"📂 모델 로드 중: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 모델 정보 확인
    input_dim = checkpoint.get('input_dim')
    if input_dim is None:
        raise ValueError("모델 파일에 input_dim 정보가 없습니다. 학습된 모델 파일을 사용해주세요.")
    
    # 모델 초기화
    model = MultiScaleEnsemble(input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 모델 로드 완료")
    print(f"   - 입력 차원: {input_dim}")
    print(f"   - 검증 정확도: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    print(f"   - 학습 에포크: {checkpoint.get('epoch', 'N/A')}")
    
    return model, input_dim

def prepare_prediction_data(data_path, window_size=60):
    """예측을 위한 데이터 준비"""
    print(f"\n📊 데이터 준비 중: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Target 컬럼 제거 (예측용이므로)
    if 'target' in df.columns:
        df = df.drop(columns=['target'])
    
    # 날짜 컬럼 확인 및 저장
    date_col = None
    if '날짜' in df.columns:
        date_col = df['날짜'].copy()
        df = df.drop(columns=['날짜'])
    
    # 최근 window_size일 데이터 추출
    if len(df) < window_size:
        raise ValueError(f"데이터가 부족합니다. 최소 {window_size}일의 데이터가 필요합니다. (현재: {len(df)}일)")
    
    # 최근 window_size일 데이터
    recent_data = df.iloc[-window_size:].values
    
    print(f"✅ 데이터 준비 완료")
    print(f"   - 전체 데이터: {len(df)}일")
    print(f"   - 예측용 데이터: {window_size}일")
    print(f"   - 피처 수: {recent_data.shape[1]}")
    
    return recent_data, date_col, df.columns.tolist()

def predict(model, data, device='cpu'):
    """모델로 예측 수행"""
    model = model.to(device)
    
    # 배치 차원 추가: (window_size, features) -> (1, window_size, features)
    data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(data_tensor)
        # 소프트맥스 확률
        probs = torch.softmax(output, dim=1)
        # 예측 클래스
        pred_class = output.argmax(dim=1).item()
        # 확률값
        prob_up = probs[0][1].item()
        prob_down = probs[0][0].item()
    
    return pred_class, prob_up, prob_down

def predict_batch(model, data_path, window_size=60, batch_size=32, device='cpu'):
    """전체 데이터에 대해 배치 예측 수행"""
    print(f"\n📊 배치 예측 시작...")
    
    df = pd.read_csv(data_path)
    
    # 날짜 컬럼 저장
    dates = None
    if '날짜' in df.columns:
        dates = df['날짜'].copy()
    
    # Target 분리
    targets = None
    if 'target' in df.columns:
        targets = df['target'].values
    
    features_df = df.drop(columns=['target'] if 'target' in df.columns else [])
    if dates is not None:
        features_df = features_df.drop(columns=['날짜'])
    
    model = model.to(device)
    model.eval()
    
    predictions = []
    probabilities_up = []
    probabilities_down = []
    
    # 슬라이딩 윈도우로 예측
    for i in range(len(features_df) - window_size):
        window_data = features_df.iloc[i:i+window_size].values
        data_tensor = torch.FloatTensor(window_data).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(data_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            prob_up = probs[0][1].item()
            prob_down = probs[0][0].item()
        
        predictions.append(pred_class)
        probabilities_up.append(prob_up)
        probabilities_down.append(prob_down)
    
    # 결과 데이터프레임 생성
    result_df = pd.DataFrame({
        '날짜': dates.iloc[window_size:].values if dates is not None else None,
        '예측': predictions,
        '상승확률': probabilities_up,
        '하락확률': probabilities_down
    })
    
    if targets is not None:
        result_df['실제'] = targets[window_size:]
        result_df['정확도'] = (result_df['예측'] == result_df['실제']).astype(int)
    
    return result_df

def main():
    parser = argparse.ArgumentParser(
        description="학습된 모델로 주식 예측 수행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 단일 예측 (최신 데이터로 내일 예측)
  python predict.py --model models/best_model_epoch_50_acc_55.23.pth --data preprocessed_005930_20240101_20241231.csv
  
  # 배치 예측 (전체 데이터에 대한 예측)
  python predict.py --model models/best_model_epoch_50_acc_55.23.pth --data preprocessed_005930_20240101_20241231.csv --batch
  
  # 결과 저장
  python predict.py --model models/best_model.pth --data preprocessed_005930.csv --batch --output predictions.csv
        """
    )
    
    parser.add_argument("--model", type=str, required=True,
                       help="모델 가중치 파일 경로 (.pth)")
    parser.add_argument("--data", type=str, required=True,
                       help="전처리된 데이터 파일 경로")
    parser.add_argument("--batch", action="store_true",
                       help="배치 예측 모드 (전체 데이터에 대해 예측)")
    parser.add_argument("--output", type=str, default=None,
                       help="결과 저장 파일 경로 (배치 모드에서만 사용)")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="사용할 디바이스 (기본값: cpu)")
    parser.add_argument("--window-size", type=int, default=60,
                       help="윈도우 크기 (기본값: 60)")
    
    args = parser.parse_args()
    
    # 파일 경로 처리
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = Path("D:/stock/_v5") / model_path
        model_path = str(model_path)
    
    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = Path("D:/stock/_v5/_data") / data_path
        data_path = str(data_path)
    
    # 디바이스 설정
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
        device = "cpu"
    
    try:
        # 모델 로드
        model, input_dim = load_model(model_path, device=device)
        
        if args.batch:
            # 배치 예측
            result_df = predict_batch(model, data_path, window_size=args.window_size, device=device)
            
            print(f"\n{'='*60}")
            print(f"📊 배치 예측 결과")
            print(f"{'='*60}")
            print(f"총 예측 수: {len(result_df)}")
            
            if '정확도' in result_df.columns:
                accuracy = result_df['정확도'].mean() * 100
                print(f"전체 정확도: {accuracy:.2f}%")
            
            print(f"\n최근 10일 예측:")
            print(result_df.tail(10).to_string(index=False))
            
            # 결과 저장
            if args.output:
                output_path = args.output
                if not os.path.isabs(output_path):
                    output_path = Path("D:/stock/_v5/_data") / output_path
                result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"\n✅ 결과 저장: {output_path}")
        else:
            # 단일 예측 (최신 데이터로 내일 예측)
            data, dates, feature_names = prepare_prediction_data(data_path, window_size=args.window_size)
            
            pred_class, prob_up, prob_down = predict(model, data, device=device)
            
            print(f"\n{'='*60}")
            print(f"🔮 예측 결과")
            print(f"{'='*60}")
            print(f"예측: {'📈 상승' if pred_class == 1 else '📉 하락'}")
            print(f"상승 확률: {prob_up*100:.2f}%")
            print(f"하락 확률: {prob_down*100:.2f}%")
            
            if dates is not None:
                print(f"\n사용된 데이터 기간:")
                print(f"  시작일: {dates.iloc[-args.window_size]}")
                print(f"  종료일: {dates.iloc[-1]}")
        
        print(f"\n{'='*60}")
        print(f"✨ 예측 완료!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

