import os
import argparse
import pandas as pd
from s01_kis_data_get import collect_stock_data
from stock_utils import StockRenamer
from s03_preprocessing import StockPreprocessor

def run_full_pipeline(code, start_date, end_date, is_train):
    base_dir = "D:/stock/_v4/_data"
    raw_path = f"{base_dir}/{code}_{start_date}_{end_date}.csv"
    renamed_path = f"{base_dir}/{code}_renamed_temp.csv"
    final_path = f"{base_dir}/preprocessed_{code}_{start_date}_{end_date}.csv"
    
    os.makedirs(base_dir, exist_ok=True)

    try:
        # 1. 데이터 수집
        print(f"\n🚀 [{code}] 데이터 수집 시작...")
        collect_stock_data(code, start_date, end_date)
        
        # 2. 리네임
        print("🔄 컬럼명 변경 중...")
        df_raw = pd.read_csv(raw_path)
        df_renamed = StockRenamer.rename(df_raw)
        df_renamed.to_csv(renamed_path, index=False, encoding='utf-8-sig')

        # 3. 전처리 (스케일러 및 날짜 처리 포함)
        print("🧪 전처리 파이프라인 가동...")
        preprocessor = StockPreprocessor()
        preprocessor.run_pipeline(renamed_path, final_path, is_train=is_train)

        # 4. 클린업 (임시 파일 삭제)
        print("🧹 임시 파일 정리 중...")
        for temp in [raw_path, renamed_path]:
            if os.path.exists(temp): os.remove(temp)
        
        print(f"\n✨ 작업 완료! 최종 파일: {final_path}")

    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="주식 데이터 수집/전처리 통합 CLI")
    parser.add_argument("--code", type=str, default="005930", help="종목코드")
    parser.add_argument("--start", type=str, required=True, help="시작일 (YYYYMMDD)")
    parser.add_argument("--end", type=str, required=True, help="종료일 (YYYYMMDD)")
    parser.add_argument("--train", action="store_true", help="스케일러 신규 학습 여부")

    args = parser.parse_args()
    run_full_pipeline(args.code, args.start, args.end, args.train)