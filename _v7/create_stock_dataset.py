import os
import argparse
import pandas as pd
from s01_kis_data_get import collect_stock_data
from s02_rename import rename_file
from s03_preprocessing import StockPreprocessor

def load_stock_codes(csv_path):
    """stockcode.csv에서 종목 코드 리스트 읽기"""
    try:
        # dtype=str로 읽어서 앞의 0이 제거되지 않도록 함
        df = pd.read_csv(csv_path, header=None, names=['code'], dtype=str)
        # 빈 줄 제거 및 공백 제거
        codes = df['code'].str.strip()
        codes = codes[codes != ''].tolist()
        # 6자리로 패딩 (앞에 0 추가)
        codes = [code.zfill(6) for code in codes]
        return codes
    except Exception as e:
        print(f"⚠️ stockcode.csv 읽기 실패: {e}")
        return []

def run_full_pipeline(code, start_date, end_date, is_train):
    # 종목 코드를 문자열로 변환하고 6자리로 패딩 (앞에 0 추가)
    code = str(code).zfill(6)
    
    base_dir = "D:/stock/_v7/_data"
    raw_path = f"{base_dir}/{code}_{start_date}_{end_date}.csv"
    renamed_path = f"{base_dir}/{code}_renamed_temp.csv"
    final_path = f"{base_dir}/preprocessed_{code}_{start_date}_{end_date}.csv"
    
    os.makedirs(base_dir, exist_ok=True)

    try:
        # 1. 데이터 수집
        print(f"\n🚀 [{code}] 데이터 수집 시작...")
        collect_stock_data(code, start_date, end_date)
        
        # 2. 리네임 (s02_rename.py의 함수 사용)
        print("\n🔄 컬럼명 변경 중...")
        if not rename_file(raw_path, renamed_path):
            raise Exception("컬럼명 변경 실패")

        # 3. 전처리 (스케일러 및 날짜 처리 포함)
        print("\n🧪 전처리 파이프라인 가동...")
        # target_epsilon을 0.005 (0.5%)로 설정
        preprocessor = StockPreprocessor(stock_code=code, target_epsilon=0.005)
        preprocessor.run_pipeline(renamed_path, final_path, is_train=is_train)

        # 4. 클린업 (임시 파일 삭제)
        print("\n🧹 임시 파일 정리 중...")
        for temp in [raw_path, renamed_path]:
            if os.path.exists(temp): os.remove(temp)
        
        print(f"\n✨ 작업 완료! 최종 파일: {final_path}")

    except Exception as e:
        print(f"⚠️ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="주식 데이터 수집/전처리 통합 CLI")
    parser.add_argument("--code", type=str, default=None, help="종목코드 (지정하지 않으면 stockcode.csv에서 읽음)")
    parser.add_argument("--start", type=str, required=True, help="시작일 (YYYYMMDD)")
    parser.add_argument("--end", type=str, required=True, help="종료일 (YYYYMMDD)")
    parser.add_argument("--train", action="store_true", help="스케일러 신규 학습 여부")
    parser.add_argument("--stockcode-file", type=str, default="D:/stock/_v7/_data/stockcode.csv", 
                        help="종목 코드 파일 경로")

    args = parser.parse_args()
    
    # 종목 코드 결정
    if args.code:
        # 단일 종목 처리
        codes = [args.code]
        print(f"📌 단일 종목 처리: {args.code}")
    else:
        # stockcode.csv에서 읽기
        codes = load_stock_codes(args.stockcode_file)
        if not codes:
            print("⚠️ 종목 코드를 찾을 수 없습니다. --code 옵션을 사용하거나 stockcode.csv 파일을 확인하세요.")
            exit(1)
        print(f"📋 총 {len(codes)}개 종목 처리 예정: {codes}")
    
    # 각 종목에 대해 파이프라인 실행
    success_count = 0
    fail_count = 0
    
    for idx, code in enumerate(codes, 1):
        print(f"\n{'='*60}")
        print(f"📊 [{idx}/{len(codes)}] 종목 코드: {code}")
        print(f"{'='*60}")
        
        try:
            run_full_pipeline(code, args.start, args.end, args.train)
            success_count += 1
            print(f"✅ [{code}] 처리 완료 ({idx}/{len(codes)})")
        except Exception as e:
            fail_count += 1
            print(f"❌ [{code}] 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 최종 요약
    print(f"\n{'='*60}")
    print(f"📈 전체 처리 완료!")
    print(f"   성공: {success_count}개")
    print(f"   실패: {fail_count}개")
    print(f"   총계: {len(codes)}개")
    print(f"{'='*60}")