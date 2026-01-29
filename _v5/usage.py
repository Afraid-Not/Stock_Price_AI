import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 모듈 임포트
from s01_kis_data_get import collect_stock_data
from s02_rename import rename_file
from s03_preprocessing import StockPreprocessor
from s05_architecture import MultiScaleEnsemble
import torch
import pandas as pd

def calculate_data_range(prediction_date_str, window_size=60, buffer_days=30):
    """예측 날짜 기준으로 필요한 데이터 수집 기간 계산"""
    # 예측 날짜 파싱
    try:
        prediction_date = datetime.strptime(prediction_date_str, '%Y%m%d')
    except ValueError:
        try:
            prediction_date = datetime.strptime(prediction_date_str, '%Y-%m-%d')
        except ValueError:
            raise ValueError("날짜 형식이 올바르지 않습니다. YYYYMMDD 또는 YYYY-MM-DD 형식을 사용하세요.")
    
    # 예측에 필요한 시작일 계산 (window_size + buffer_days 전)
    # 주말/공휴일을 고려하여 여유있게 계산
    total_days_needed = window_size + buffer_days
    start_date = prediction_date - timedelta(days=int(total_days_needed * 1.5))  # 여유있게 1.5배
    
    # 종료일은 예측일 하루 전 (예측일 당일 데이터는 아직 없으므로)
    end_date = prediction_date - timedelta(days=1)
    
    # 주말 제외하여 평일로 조정
    while end_date.weekday() >= 5:  # 토요일(5) 또는 일요일(6)
        end_date -= timedelta(days=1)
    
    return start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d')

def load_model(model_path, device='cpu'):
    """저장된 모델 가중치 로드"""
    print(f"📂 모델 로드 중: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    input_dim = checkpoint.get('input_dim')
    if input_dim is None:
        raise ValueError("모델 파일에 input_dim 정보가 없습니다.")
    
    model = MultiScaleEnsemble(input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 모델 로드 완료 (입력 차원: {input_dim})")
    return model, input_dim

def prepare_prediction_data(data_path, window_size=60):
    """예측을 위한 데이터 준비"""
    df = pd.read_csv(data_path)
    
    # Target 컬럼 제거
    if 'target' in df.columns:
        df = df.drop(columns=['target'])
    
    # 날짜 컬럼 제거
    if '날짜' in df.columns:
        df = df.drop(columns=['날짜'])
    
    # 최근 window_size일 데이터 추출
    if len(df) < window_size:
        raise ValueError(f"데이터가 부족합니다. 최소 {window_size}일의 데이터가 필요합니다.")
    
    recent_data = df.iloc[-window_size:].values
    return recent_data

def predict(model, data, device='cpu'):
    """모델로 예측 수행"""
    model = model.to(device)
    data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(data_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        prob_up = probs[0][1].item()
        prob_down = probs[0][0].item()
    
    return pred_class, prob_up, prob_down

def run_prediction_pipeline(code, prediction_date, model_path, window_size=60, device='cpu'):
    """전체 파이프라인: 데이터 수집 → 전처리 → 예측"""
    base_dir = "D:/stock/_v5/_data"
    os.makedirs(base_dir, exist_ok=True)
    
    # 디바이스 설정
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
        device = "cpu"
    
    try:
        # 1. 필요한 데이터 기간 계산
        print(f"\n{'='*60}")
        print(f"📅 예측 날짜: {prediction_date}")
        print(f"{'='*60}")
        
        start_date, end_date = calculate_data_range(prediction_date, window_size=window_size)
        print(f"📊 데이터 수집 기간: {start_date} ~ {end_date}")
        
        raw_path = f"{base_dir}/{code}_{start_date}_{end_date}.csv"
        renamed_path = f"{base_dir}/{code}_renamed_temp.csv"
        final_path = f"{base_dir}/preprocessed_{code}_{start_date}_{end_date}.csv"
        
        # 2. 데이터 수집
        print(f"\n{'='*60}")
        print(f"📥 1단계: 데이터 수집")
        print(f"{'='*60}")
        
        # 이미 수집된 데이터가 있으면 확인
        if os.path.exists(raw_path):
            print(f"✅ 기존 데이터 파일 발견: {raw_path}")
            response = input("다시 수집하시겠습니까? (y/N): ").strip().lower()
            if response == 'y':
                collect_stock_data(code, start_date, end_date)
        else:
            collect_stock_data(code, start_date, end_date)
        
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"데이터 수집 실패: {raw_path}")
        
        # 3. 전처리
        print(f"\n{'='*60}")
        print(f"🔄 2단계: 데이터 전처리")
        print(f"{'='*60}")
        
        # 이미 전처리된 파일이 있으면 확인
        if os.path.exists(final_path):
            print(f"✅ 기존 전처리 파일 발견: {final_path}")
            response = input("다시 전처리하시겠습니까? (y/N): ").strip().lower()
            if response == 'y':
                # 컬럼명 변경
                print("📝 컬럼명 변경 중...")
                if not rename_file(raw_path, renamed_path):
                    raise Exception("컬럼명 변경 실패")
                
                # 전처리
                print("🧪 전처리 파이프라인 실행 중...")
                preprocessor = StockPreprocessor()
                preprocessor.run_pipeline(renamed_path, final_path, is_train=False)
                
                # 임시 파일 정리
                if os.path.exists(renamed_path):
                    os.remove(renamed_path)
        else:
            # 컬럼명 변경
            print("📝 컬럼명 변경 중...")
            if not rename_file(raw_path, renamed_path):
                raise Exception("컬럼명 변경 실패")
            
            # 전처리
            print("🧪 전처리 파이프라인 실행 중...")
            preprocessor = StockPreprocessor()
            preprocessor.run_pipeline(renamed_path, final_path, is_train=False)
            
            # 임시 파일 정리
            if os.path.exists(renamed_path):
                os.remove(renamed_path)
        
        # 4. 모델 로드
        print(f"\n{'='*60}")
        print(f"📂 3단계: 모델 로드")
        print(f"{'='*60}")
        
        model, input_dim = load_model(model_path, device=device)
        
        # 5. 예측 데이터 준비
        print(f"\n{'='*60}")
        print(f"📊 4단계: 예측 데이터 준비")
        print(f"{'='*60}")
        
        prediction_data = prepare_prediction_data(final_path, window_size=window_size)
        print(f"✅ 데이터 준비 완료 (최근 {window_size}일 데이터 사용)")
        
        # 6. 예측 수행
        print(f"\n{'='*60}")
        print(f"🔮 5단계: 예측 수행")
        print(f"{'='*60}")
        
        pred_class, prob_up, prob_down = predict(model, prediction_data, device=device)
        
        # 결과 출력
        print(f"\n{'='*60}")
        print(f"✨ 예측 결과")
        print(f"{'='*60}")
        
        # 예측 날짜 포맷팅
        try:
            pred_date = datetime.strptime(prediction_date, '%Y%m%d')
        except:
            try:
                pred_date = datetime.strptime(prediction_date, '%Y-%m-%d')
            except:
                pred_date = None
        
        if pred_date:
            pred_date_str = pred_date.strftime('%Y-%m-%d')
            print(f"📅 예측 대상 날짜: {pred_date_str}")
        
        print(f"\n예측: {'📈 상승' if pred_class == 1 else '📉 하락'}")
        print(f"상승 확률: {prob_up*100:.2f}%")
        print(f"하락 확률: {prob_down*100:.2f}%")
        
        print(f"\n{'='*60}")
        print(f"✅ 예측 완료!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="주식 예측 간편 사용 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용 (내일 예측)
  python usage.py --code 005930 --date 20260128
  
  # 특정 날짜 예측
  python usage.py --code 005930 --date 2026-01-28
  
  # GPU 사용
  python usage.py --code 005930 --date 20260128 --device cuda
        """
    )
    
    parser.add_argument("--code", type=str, required=True,
                       help="종목코드 (예: 005930)")
    parser.add_argument("--date", type=str, required=True,
                       help="예측하고자 하는 날짜 (YYYYMMDD 또는 YYYY-MM-DD)")
    parser.add_argument("--model", type=str, default="models/best_model_epoch_38_acc_50.18_f1_0.5345.pth",
                       help="모델 가중치 파일 경로 (기본값: models/best_model_epoch_68_acc_54.04.pth)")
    parser.add_argument("--window-size", type=int, default=60,
                       help="예측 윈도우 크기 (기본값: 60)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="사용할 디바이스 (기본값: cpu)")
    
    args = parser.parse_args()
    
    # 모델 경로 처리
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = Path("D:/stock/_v5") / model_path
        model_path = str(model_path)
    
    # 날짜 형식 정규화 (YYYYMMDD로 변환)
    prediction_date = args.date.replace('-', '').replace('/', '')
    if len(prediction_date) != 8:
        print("❌ 날짜 형식이 올바르지 않습니다. YYYYMMDD 또는 YYYY-MM-DD 형식을 사용하세요.")
        sys.exit(1)
    
    run_prediction_pipeline(
        code=args.code,
        prediction_date=prediction_date,
        model_path=model_path,
        window_size=args.window_size,
        device=args.device
    )

if __name__ == "__main__":
    main()

