import pandas as pd
import argparse
import os
import glob
from pathlib import Path
from stock_utils import StockRenamer

def rename_file(input_file, output_file=None):
    if output_file is None:
        input_path = Path(input_file)
        if '_renamed' not in input_path.stem:
            output_file = input_path.parent / f"{input_path.stem}_renamed{input_path.suffix}"
        else:
            output_file = input_path
    
    try:
        df = pd.read_csv(input_file)
        df_renamed = StockRenamer.rename(df)
        
        # 순매수 검증 (핵심 투자자만)
        for inv in ['개인', '외국인', '기관계']:
            if all(f'{inv}_{suffix}' in df_renamed.columns for suffix in ['매수수량', '매도수량', '순매수수량']):
                diff = (df_renamed[f'{inv}_매수수량'] - df_renamed[f'{inv}_매도수량'] - df_renamed[f'{inv}_순매수수량']).abs().max()
                if diff > 1: print(f"⚠️ {inv} 수량 검증 주의 (차이: {diff})")
        
        df_renamed.to_csv(str(output_file), index=False, encoding='utf-8-sig')
        print(f"💾 저장 완료: {output_file}")
        return True
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", nargs="+", required=True)
    parser.add_argument("--output", "-o")
    args = parser.parse_args()
    
    for pattern in args.input:
        for f in glob.glob(pattern) if '*' in pattern else [pattern]:
            if os.path.exists(f): rename_file(f, args.output)

if __name__ == "__main__":
    main()