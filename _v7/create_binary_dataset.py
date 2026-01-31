import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def convert_to_binary(input_path, output_path, target_epsilon=0.005):
    """
    기존 3진 분류 데이터를 이진 분류로 변환 (보합 제거)
    
    Args:
        input_path: 입력 CSV 파일 경로 (3진 분류)
        output_path: 출력 CSV 파일 경로 (이진 분류)
        target_epsilon: 보합 범위 (기본값 0.005 = 0.5%)
    """
    import sys
    import io
    
    # Windows 콘솔 인코딩 설정
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print(f"[데이터 로드] {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"   원본 데이터: {len(df)}행")
    print(f"   원본 타겟 분포: {df['target'].value_counts().sort_index().to_dict()}")
    
    # 보합(1) 데이터 제거
    before_remove = len(df)
    df = df[df['target'] != 1].copy()  # 보합 제거
    after_remove = len(df)
    removed_count = before_remove - after_remove
    
    print(f"   보합 데이터 제거: {removed_count}행 제거됨 ({before_remove}행 -> {after_remove}행)")
    
    # 타겟 재매핑: 0(하락) -> 0, 2(상승) -> 1
    df['target'] = df['target'].map({0: 0, 2: 1})
    
    # 타겟 분포 확인
    target_counts = df['target'].value_counts().sort_index()
    print(f"   최종 타겟 분포: {dict(target_counts)} (0:하락, 1:상승)")
    
    # 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"[완료] 이진 분류 데이터 저장: {output_path} ({len(df)}행)")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3진 분류 데이터를 이진 분류로 변환")
    parser.add_argument("--input", type=str, 
                       default="D:/stock/_v7/_data/preprocessed_005930_20100101_20260129.csv",
                       help="입력 CSV 파일 경로 (3진 분류)")
    parser.add_argument("--output", type=str,
                       default="D:/stock/_v7/_data/preprocessed_binary_005930_20100101_20260129.csv",
                       help="출력 CSV 파일 경로 (이진 분류)")
    
    args = parser.parse_args()
    
    convert_to_binary(args.input, args.output)

