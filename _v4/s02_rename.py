import pandas as pd
from stock_utils import StockRenamer  # 모듈 임포트

def main():
    input_file = "D:/stock/_v3/_data/005930_merged.csv"
    output_file = "D:/stock/_v3/_data/005930_renamed.csv"

    # 1. 파일 읽기
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {input_file}")
        return

    # 2. 모듈을 사용하여 리네임 수행
    df_renamed = StockRenamer.rename(df)

    # 3. 저장
    df_renamed.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"처리 완료: {df_renamed.shape}")
    print(f"결과 저장: {output_file}")

if __name__ == "__main__":
    main()