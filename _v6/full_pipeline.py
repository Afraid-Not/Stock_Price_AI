import os
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from s01_kis_data_get import collect_stock_data
from s02_rename import rename_file
from s03_preprocessing import StockPreprocessor
from predict import StockPredictor

def date_to_str(date_obj):
    """날짜 객체를 YYYYMMDD 문자열로 변환"""
    return date_obj.strftime("%Y%m%d")

def str_to_date(date_str):
    """YYYYMMDD 문자열을 날짜 객체로 변환"""
    return datetime.strptime(date_str, "%Y%m%d")

def run_full_pipeline(code, date_str, tomorrow_str, model_dir="D:/stock/_v6/models", scaler_dir="D:/stock/_v6/scalers"):
    """
    전체 파이프라인 실행:
    1. 데이터 수집
    2. 전처리
    3. 예측
    4. 결과 출력
    """
    base_dir = Path("D:/stock/_v6/_data")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # 날짜 파싱
    try:
        date = str_to_date(date_str)
        tomorrow = str_to_date(tomorrow_str)
    except ValueError as e:
        print(f"❌ 날짜 형식 오류: {e}")
        print("   날짜는 YYYYMMDD 형식이어야 합니다.")
        return
    
    # 데이터 수집 기간 설정 (과거 데이터를 충분히 수집하기 위해)
    # 최소 30일 이상의 과거 데이터 필요
    start_date = date - timedelta(days=60)  # 충분한 과거 데이터 수집
    start_date_str = date_to_str(start_date)
    end_date_str = date_to_str(date)  # 오늘까지
    
    print(f"\n{'='*60}")
    print(f"🚀 Full Pipeline Started")
    print(f"{'='*60}")
    print(f"Stock Code:     {code}")
    print(f"Date:           {date_str}")
    print(f"Tomorrow:       {tomorrow_str}")
    print(f"Data Range:     {start_date_str} ~ {end_date_str}")
    print(f"{'='*60}")
    
    # 파일 경로 설정
    raw_path = base_dir / f"{code}_{start_date_str}_{end_date_str}.csv"
    renamed_path = base_dir / f"{code}_renamed_temp_{date_str}.csv"
    preprocessed_path = base_dir / f"preprocessed_{code}_{start_date_str}_{end_date_str}.csv"
    
    try:
        # ========== 1. 데이터 수집 ==========
        print(f"\n{'='*60}")
        print(f"📥 Step 1: Data Collection")
        print(f"{'='*60}")
        print(f"[{code}] 데이터 수집 시작...")
        collect_stock_data(code, start_date_str, end_date_str)
        
        if not raw_path.exists():
            raise FileNotFoundError(f"데이터 수집 실패: {raw_path}")
        
        # ========== 2. 컬럼명 변경 ==========
        print(f"\n{'='*60}")
        print(f"🔄 Step 2: Column Renaming")
        print(f"{'='*60}")
        if not rename_file(str(raw_path), str(renamed_path)):
            raise Exception("컬럼명 변경 실패")
        
        # ========== 3. 전처리 ==========
        print(f"\n{'='*60}")
        print(f"🧪 Step 3: Preprocessing")
        print(f"{'='*60}")
        preprocessor = StockPreprocessor(scaler_dir=scaler_dir, stock_code=code)
        preprocessor.run_pipeline(str(renamed_path), str(preprocessed_path), is_train=False)
        
        # 전처리된 파일은 예측 후 삭제할 예정이므로 여기서는 생성만 함
        
        # ========== 4. 원본 데이터에서 현재 종가 가져오기 ==========
        print(f"\n{'='*60}")
        print(f"💰 Step 4: Get Current Price")
        print(f"{'='*60}")
        original_df = pd.read_csv(raw_path)
        if '종가' in original_df.columns:
            current_price = original_df['종가'].iloc[-1]
        elif 'stck_clpr' in original_df.columns:
            current_price = original_df['stck_clpr'].iloc[-1]
        else:
            print("⚠️ 종가 정보를 찾을 수 없습니다. 수동으로 입력해주세요.")
            current_price = None
        
        if current_price:
            print(f"Current Price: {current_price:,.0f} KRW")
        
        # ========== 5. 예측 ==========
        print(f"\n{'='*60}")
        print(f"🔮 Step 5: Prediction")
        print(f"{'='*60}")
        
        # 종목코드별 모델 폴더 경로 확인
        model_path = Path(model_dir) / code
        print(f"📁 모델 검색 경로: {model_path}")
        if not model_path.exists():
            print(f"⚠️ 종목코드 폴더가 없습니다: {model_path}")
            print(f"   먼저 train.py로 모델을 학습시켜주세요.")
        
        predictor = StockPredictor(model_dir=model_dir, scaler_dir=scaler_dir, stock_code=code)
        predictor.load_models(model_name=None, stock_code=code)  # 해당 종목의 최신 모델 사용
        
        # 데이터 로드
        X, df, _ = predictor.load_data(str(preprocessed_path), str(raw_path))
        
        # 예측 수행
        ensemble_pred, lgbm_pred = predictor.predict(X)
        
        # 스케일러 역변환
        predictions_original = predictor.inverse_transform(ensemble_pred)
        lgbm_pred_original = predictor.inverse_transform(lgbm_pred)
        
        # ========== 6. 결과 출력 ==========
        print(f"\n{'='*60}")
        print(f"📊 Step 6: Results")
        print(f"{'='*60}")
        
        if len(predictions_original) == 0:
            print("❌ 예측 결과가 없습니다.")
            return
        
        # 마지막 예측 결과 (가장 최근 데이터)
        last_prediction = predictions_original[-1]
        direction = "상승" if last_prediction > 0 else "하락"
        
        print(f"\n{'='*60}")
        print(f"🎯 Prediction Result")
        print(f"{'='*60}")
        print(f"Date:              {date_str}")
        print(f"Tomorrow:          {tomorrow_str}")
        print(f"Stock Code:        {code}")
        print(f"\nPredicted Return:   {last_prediction*100:.2f}%")
        print(f"Direction:         {direction}")
        
        if current_price:
            next_price, min_price, max_price = predictor.calculate_next_day_price(
                current_price,
                last_prediction,
                lgbm_pred_original[-1]
            )
            
            print(f"\n💰 Price Prediction:")
            print(f"Current Price:     {current_price:,.0f} KRW")
            print(f"Expected Price:    {next_price:,.0f} KRW")
            print(f"Price Range:       {min_price:,.0f} ~ {max_price:,.0f} KRW")
            
            print(f"\n{'='*60}")
            print(f"📈 Summary")
            print(f"{'='*60}")
            print(f"현재가: {current_price:,.0f}원")
            print(f"예상가: {next_price:,.0f}원 ({direction})")
            print(f"예상 범위: {min_price:,.0f}원 ~ {max_price:,.0f}원")
            print(f"예상 수익률: {last_prediction*100:.2f}%")
        else:
            print(f"\n⚠️ 현재 종가 정보가 없어 가격 예측을 수행할 수 없습니다.")
        
        # ========== 7. 클린업 ==========
        print(f"\n{'='*60}")
        print(f"🧹 Step 7: Cleanup")
        print(f"{'='*60}")
        
        # 임시 파일 삭제 (원본, 리네임된 파일, 전처리된 파일 모두 삭제)
        temp_files = [raw_path, renamed_path, preprocessed_path]
        for temp_file in temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                    print(f"   Deleted: {temp_file.name}")
                except Exception as e:
                    print(f"   ⚠️ Could not delete {temp_file.name}: {e}")
        
        print(f"\n✅ Full pipeline completed successfully!")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return

def main():
    parser = argparse.ArgumentParser(
        description="주식 데이터 수집 → 전처리 → 예측 전체 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 오늘 날짜 기준으로 예측
  python full_pipeline.py --code 005930 --date 20260128 --tomorrow 20260129
  
  # 특정 날짜 기준으로 예측
  python full_pipeline.py --code 005930 --date 20260127 --tomorrow 20260128
        """
    )
    
    parser.add_argument("--code", type=str, required=True, help="종목코드 (예: 005930)")
    parser.add_argument("--date", type=str, required=True, help="오늘 날짜 (YYYYMMDD)")
    parser.add_argument("--tomorrow", type=str, required=True, help="내일 날짜 (YYYYMMDD)")
    parser.add_argument("--model_dir", type=str, default="D:/stock/_v6/models", help="모델 디렉토리")
    parser.add_argument("--scaler_dir", type=str, default="D:/stock/_v6/scalers", help="스케일러 디렉토리")
    
    args = parser.parse_args()
    
    run_full_pipeline(
        code=args.code,
        date_str=args.date,
        tomorrow_str=args.tomorrow,
        model_dir=args.model_dir,
        scaler_dir=args.scaler_dir
    )

if __name__ == "__main__":
    main()

