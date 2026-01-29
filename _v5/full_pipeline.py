import argparse
import os
import sys
import torch
import pandas as pd
from pathlib import Path

# 모듈 임포트
from s00_get_token import get_access_token  # 토큰 발급 (s01에서 사용)
from s01_kis_data_get import collect_stock_data
from s02_rename import rename_file
from s03_preprocessing import StockPreprocessor
from s05_architecture import MultiScaleEnsemble

def collect_data(code, start_date, end_date, base_dir):
    """1단계: 주식 데이터 수집"""
    print(f"\n{'='*60}")
    print(f"📊 1단계: 데이터 수집 시작")
    print(f"{'='*60}")
    print(f"종목코드: {code}")
    print(f"기간: {start_date} ~ {end_date}")
    
    raw_path = f"{base_dir}/{code}_{start_date}_{end_date}.csv"
    
    # 이미 수집된 데이터가 있으면 스킵 옵션 제공
    if os.path.exists(raw_path):
        print(f"⚠️ 이미 수집된 데이터가 있습니다: {raw_path}")
        response = input("다시 수집하시겠습니까? (y/N): ").strip().lower()
        if response != 'y':
            print("✅ 기존 데이터 사용")
            return raw_path
    
    collect_stock_data(code, start_date, end_date)
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"데이터 수집 실패: {raw_path}")
    
    print(f"✅ 데이터 수집 완료: {raw_path}")
    return raw_path

def preprocess_data(raw_path, code, start_date, end_date, base_dir, is_train):
    """2단계: 데이터 전처리"""
    print(f"\n{'='*60}")
    print(f"🔄 2단계: 데이터 전처리")
    print(f"{'='*60}")
    
    renamed_path = f"{base_dir}/{code}_renamed_temp.csv"
    final_path = f"{base_dir}/preprocessed_{code}_{start_date}_{end_date}.csv"
    
    # 컬럼명 변경
    print("📝 컬럼명 변경 중...")
    if not rename_file(raw_path, renamed_path):
        raise Exception("컬럼명 변경 실패")
    
    # 전처리
    print("🧪 전처리 파이프라인 실행 중...")
    preprocessor = StockPreprocessor()
    preprocessor.run_pipeline(renamed_path, final_path, is_train=is_train)
    
    # 임시 파일 정리
    if os.path.exists(renamed_path):
        os.remove(renamed_path)
    
    print(f"✅ 전처리 완료: {final_path}")
    return final_path

def load_model(model_path, device='cpu'):
    """저장된 모델 가중치 로드"""
    print(f"\n{'='*60}")
    print(f"📂 3단계: 모델 로드")
    print(f"{'='*60}")
    
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

def prepare_prediction_data(data_path, window_size=60, code=None, end_date=None):
    """예측을 위한 데이터 준비"""
    print(f"\n{'='*60}")
    print(f"📊 4단계: 예측 데이터 준비")
    print(f"{'='*60}")
    
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
    else:
        # 날짜 컬럼이 없으면 원본 파일에서 날짜 가져오기 시도
        # 파일명에서 정보 추출: preprocessed_005930_20080701_20260127.csv
        import re
        from datetime import datetime, timedelta
        filename = os.path.basename(data_path)
        date_match = re.search(r'preprocessed_(\d{6})_(\d{8})_(\d{8})', filename)
        if date_match:
            code_str = date_match.group(1)
            start_date_str = date_match.group(2)
            end_date_str = date_match.group(3)
            
            # 원본 파일 경로 시도
            base_dir = os.path.dirname(data_path)
            raw_path = f"{base_dir}/{code_str}_{start_date_str}_{end_date_str}.csv"
            renamed_path = f"{base_dir}/{code_str}_renamed_temp.csv"
            
            # 리네임된 파일이나 원본 파일에서 날짜 읽기
            for try_path in [renamed_path, raw_path]:
                if os.path.exists(try_path):
                    try:
                        raw_df = pd.read_csv(try_path)
                        if '날짜' in raw_df.columns:
                            # 날짜 형식 변환
                            raw_df['날짜'] = pd.to_datetime(raw_df['날짜'].astype(str), format='%Y%m%d', errors='coerce')
                            raw_df = raw_df.dropna(subset=['날짜'])
                            # 전처리된 데이터와 행 수가 맞는지 확인
                            if len(raw_df) >= len(df):
                                # 마지막 len(df)개만 사용
                                date_col = raw_df['날짜'].iloc[-len(df):].reset_index(drop=True)
                                date_col = date_col.dt.strftime('%Y-%m-%d')
                                break
                    except Exception as e:
                        continue
            
            # 파일에서 읽지 못했으면 파일명의 종료일 기준으로 추정
            if date_col is None:
                try:
                    end_date = datetime.strptime(end_date_str, '%Y%m%d')
                    # 주말 제외하고 날짜 생성
                    dates = []
                    current_date = end_date
                    for _ in range(len(df)):
                        # 주말이면 평일로 이동
                        while current_date.weekday() >= 5:  # 토요일(5) 또는 일요일(6)
                            current_date -= timedelta(days=1)
                        dates.insert(0, current_date.strftime('%Y-%m-%d'))
                        current_date -= timedelta(days=1)
                    date_col = pd.Series(dates)
                except:
                    pass
    
    # 최근 window_size일 데이터 추출
    if len(df) < window_size:
        raise ValueError(f"데이터가 부족합니다. 최소 {window_size}일의 데이터가 필요합니다. (현재: {len(df)}일)")
    
    # 최근 window_size일 데이터
    recent_data = df.iloc[-window_size:].values
    
    print(f"✅ 데이터 준비 완료")
    print(f"   - 전체 데이터: {len(df)}일")
    print(f"   - 예측용 데이터: {window_size}일")
    print(f"   - 피처 수: {recent_data.shape[1]}")
    
    return recent_data, date_col

def predict(model, data, device='cpu'):
    """모델로 예측 수행"""
    print(f"\n{'='*60}")
    print(f"🔮 5단계: 예측 수행")
    print(f"{'='*60}")
    
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

def run_full_pipeline(code, model_path, start_date=None, end_date=None,
                     skip_collect=False, skip_preprocess=False,
                     is_train=False, window_size=60, device='cpu', predict_tomorrow=False,
                     data_path=None):
    """전체 파이프라인 실행: 데이터 수집 → 전처리 → 예측"""
    base_dir = "D:/stock/_v5/_data"
    os.makedirs(base_dir, exist_ok=True)
    
    # 데이터 경로 결정
    if data_path:
        # 직접 지정된 파일 경로 사용
        final_path = data_path
        if not os.path.isabs(final_path):
            final_path = os.path.join(base_dir, final_path)
    elif start_date and end_date:
        # 날짜로 파일 경로 생성
        raw_path = f"{base_dir}/{code}_{start_date}_{end_date}.csv"
        final_path = f"{base_dir}/preprocessed_{code}_{start_date}_{end_date}.csv"
    else:
        raise ValueError("--data 옵션 또는 --start/--end 옵션이 필요합니다.")
    
    # 디바이스 설정
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA를 사용할 수 없습니다. CPU로 전환합니다.")
        device = "cpu"
    
    try:
        # 1. 데이터 수집 (s00, s01)
        if not skip_collect:
            raw_path = collect_data(code, start_date, end_date, base_dir)
        
        # 2. 전처리 (s02, s03)
        if not skip_preprocess:
            if data_path:
                raise ValueError("--data 옵션을 사용할 때는 --skip-preprocess 옵션도 함께 사용해야 합니다.")
            # skip_collect가 True면 원본 파일이 없을 수 있으므로 확인
            if skip_collect and not os.path.exists(raw_path):
                raise FileNotFoundError(f"원본 데이터 파일이 없습니다: {raw_path}\n전처리 단계를 실행하려면 원본 파일이 필요합니다.")
            final_path = preprocess_data(raw_path, code, start_date, end_date, base_dir, is_train)
        else:
            # 전처리 스킵 시 전처리된 파일 확인
            if not os.path.exists(final_path):
                raise FileNotFoundError(f"전처리 단계를 건너뛰었지만 전처리된 파일이 없습니다: {final_path}")
        
        # 3. 모델 로드 (s05)
        model, input_dim = load_model(model_path, device=device)
        
        # 4. 예측 데이터 준비 (s04 개념 활용)
        prediction_data, date_col = prepare_prediction_data(final_path, window_size=window_size, code=code, end_date=end_date if end_date else None)
        
        # 5. 예측 수행 (s05)
        pred_class, prob_up, prob_down = predict(model, prediction_data, device=device)
        
        # 결과 출력
        print(f"\n{'='*60}")
        print(f"✨ 예측 결과")
        print(f"{'='*60}")
        
        # 예측 대상 날짜 계산
        prediction_date_str = None
        last_date_str = None
        from datetime import datetime, timedelta
        
        if predict_tomorrow:
            # --tomorrow 옵션: 오늘 날짜 기준으로 내일 예측
            today = datetime.now()
            # 다음 거래일 계산 (주말 제외)
            tomorrow = today + timedelta(days=1)
            if tomorrow.weekday() == 5:  # 토요일
                tomorrow += timedelta(days=2)
            elif tomorrow.weekday() == 6:  # 일요일
                tomorrow += timedelta(days=1)
            
            prediction_date_str = tomorrow.strftime('%Y-%m-%d')
            print(f"📅 예측 대상 날짜: {prediction_date_str} (내일)")
            print(f"📊 기준 날짜: {today.strftime('%Y-%m-%d')} (오늘)")
        elif date_col is not None and len(date_col) > 0:
            last_date_str = str(date_col.iloc[-1])
            # 날짜 형식 파싱 (YYYY-MM-DD 또는 YYYYMMDD 등)
            try:
                # 다양한 날짜 형식 시도
                last_date = None
                for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
                    try:
                        last_date = datetime.strptime(last_date_str, fmt)
                        break
                    except ValueError:
                        continue
                
                if last_date:
                    # 다음 거래일 예측 (주말 제외)
                    next_date = last_date + timedelta(days=1)
                    # 토요일이면 월요일로
                    if next_date.weekday() == 5:  # 토요일
                        next_date += timedelta(days=2)
                    elif next_date.weekday() == 6:  # 일요일
                        next_date += timedelta(days=1)
                    
                    prediction_date_str = next_date.strftime('%Y-%m-%d')
                    print(f"📅 예측 대상 날짜: {prediction_date_str} (다음 거래일)")
                    print(f"📊 기준 날짜: {last_date.strftime('%Y-%m-%d')} (마지막 데이터)")
            except Exception as e:
                pass
        
        print(f"\n예측: {'📈 상승' if pred_class == 1 else '📉 하락'}")
        print(f"상승 확률: {prob_up*100:.2f}%")
        print(f"하락 확률: {prob_down*100:.2f}%")
        
        if date_col is not None and len(date_col) > 0:
            print(f"\n사용된 데이터 기간:")
            print(f"  시작일: {date_col.iloc[-window_size]}")
            print(f"  종료일: {date_col.iloc[-1]}")
            if prediction_date_str:
                print(f"  예측일: {prediction_date_str}")
        
        print(f"\n{'='*60}")
        print(f"✅ 전체 파이프라인 완료!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="주식 데이터 수집/전처리/예측 통합 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 파이프라인 실행 (수집 → 전처리 → 예측)
  python full_pipeline.py --code 005930 --start 20240101 --end 20241231 --model models/best_model.pth
  
  # 전처리된 데이터로 바로 예측 (수집/전처리 스킵)
  python full_pipeline.py --code 005930 --start 20240101 --end 20241231 --model models/best_model.pth --skip-collect --skip-preprocess
  
  # 스케일러 재학습 후 예측
  python full_pipeline.py --code 005930 --start 20240101 --end 20241231 --model models/best_model.pth --train
  
  # CUDA 사용
  python full_pipeline.py --code 005930 --start 20240101 --end 20241231 --model models/best_model.pth --device cuda
  
  # 내일 예측 (오늘 날짜 기준) - 파일 직접 지정
  python full_pipeline.py --model models/best_model.pth --data preprocessed_005930_20080701_20260127.csv --tomorrow
  
  # 내일 예측 (오늘 날짜 기준) - 날짜로 파일 찾기
  python full_pipeline.py --code 005930 --start 20080701 --end 20260127 --model models/best_model.pth --tomorrow --skip-collect --skip-preprocess
        """
    )
    
    # 필수 인자
    parser.add_argument("--code", type=str, help="종목코드 (예: 005930) - 데이터 수집/전처리 시 필요")
    parser.add_argument("--start", type=str, help="시작일 (YYYYMMDD) - 데이터 수집/전처리 시 필요")
    parser.add_argument("--end", type=str, help="종료일 (YYYYMMDD) - 데이터 수집/전처리 시 필요")
    parser.add_argument("--model", type=str, required=True, help="모델 가중치 파일 경로 (.pth)")
    parser.add_argument("--data", type=str, help="전처리된 데이터 파일 경로 (--skip-collect --skip-preprocess 사용 시)")
    
    # 단계 스킵 옵션
    parser.add_argument("--skip-collect", action="store_true", help="데이터 수집 단계 건너뛰기")
    parser.add_argument("--skip-preprocess", action="store_true", help="전처리 단계 건너뛰기")
    
    # 전처리 옵션
    parser.add_argument("--train", action="store_true", help="스케일러 신규 학습 (기본값: False)")
    
    # 예측 옵션
    parser.add_argument("--window-size", type=int, default=60, help="예측 윈도우 크기 (기본값: 60)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="사용할 디바이스 (기본값: cpu)")
    parser.add_argument("--tomorrow", action="store_true",
                       help="오늘 날짜 기준으로 내일 예측 (기본값: 마지막 데이터의 다음 거래일)")
    
    args = parser.parse_args()
    
    # 인자 검증
    if args.data:
        # --data 옵션 사용 시
        if not args.skip_preprocess:
            print("❌ --data 옵션을 사용할 때는 --skip-preprocess 옵션도 필요합니다.")
            sys.exit(1)
        code = args.code  # 파일명에서 추출 가능하지만 일단 None 허용
        start_date = None
        end_date = None
    else:
        # --start/--end 옵션 사용 시
        if not args.code or not args.start or not args.end:
            print("❌ --data 옵션을 사용하지 않을 경우 --code, --start, --end가 모두 필요합니다.")
            sys.exit(1)
        
        # 날짜 형식 검증
        try:
            from datetime import datetime
            datetime.strptime(args.start, "%Y%m%d")
            datetime.strptime(args.end, "%Y%m%d")
        except ValueError:
            print("❌ 날짜 형식이 올바르지 않습니다. YYYYMMDD 형식으로 입력해주세요.")
            sys.exit(1)
        
        code = args.code
        start_date = args.start
        end_date = args.end
    
    # 모델 경로 처리
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = Path("D:/stock/_v5") / model_path
        model_path = str(model_path)
    
    run_full_pipeline(
        code=code,
        model_path=model_path,
        start_date=start_date,
        end_date=end_date,
        skip_collect=args.skip_collect,
        skip_preprocess=args.skip_preprocess,
        is_train=args.train,
        window_size=args.window_size,
        device=args.device,
        predict_tomorrow=args.tomorrow,
        data_path=args.data
    )

if __name__ == "__main__":
    main()