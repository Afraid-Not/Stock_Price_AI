import argparse
from s01_kis_data_get import collect_stock_data
from s02_rename import rename_file
from s03_preprocessing import StockPreprocessor

def run_full_pipeline(code, start, end, is_train):
    code = str(code).zfill(6)
    base_dir = "D:/stock/_v9/_data"
    
    raw_file = f"{base_dir}/{code}_{start}_{end}.csv"
    renamed_file = f"{base_dir}/{code}_renamed.csv"
    final_file = f"{base_dir}/preprocessed_{code}_{start}_{end}.csv"
    
    # 실행
    collect_stock_data(code, start, end)
    if rename_file(raw_file, renamed_file):
        preprocessor = StockPreprocessor(stock_code=code)
        preprocessor.run_pipeline(renamed_file, final_file, is_train=is_train)
        
    print(f"✨ [{code}] 파이프라인 완료")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    args = parser.parse_args()
    
    run_full_pipeline(args.code, args.start, args.end, True)