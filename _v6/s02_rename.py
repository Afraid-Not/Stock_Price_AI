import pandas as pd
import argparse
import os
import glob
from pathlib import Path
from stock_utils import StockRenamer  # 모듈 임포트

def rename_file(input_file, output_file=None):
    """
    단일 파일의 컬럼명을 변경합니다.
    
    Args:
        input_file: 입력 파일 경로
        output_file: 출력 파일 경로 (None이면 자동 생성)
    """
    # 출력 파일 경로가 지정되지 않으면 자동 생성
    if output_file is None:
        input_path = Path(input_file)
        # 파일명에 _renamed가 없으면 추가, 있으면 그대로 사용
        if '_renamed' not in input_path.stem:
            output_file = input_path.parent / f"{input_path.stem}_renamed{input_path.suffix}"
        else:
            output_file = input_path
    
    # 1. 파일 읽기
    try:
        df = pd.read_csv(input_file)
        print(f"📂 파일 읽기 완료: {input_file} ({len(df)}행)")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
        return False
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return False

    # 2. 모듈을 사용하여 리네임 수행
    try:
        df_renamed = StockRenamer.rename(df)
        print(f"🔄 컬럼명 변경 완료: {len(df_renamed.columns)}개 컬럼")
        
        # 3. 순매수 검증 (매수 - 매도 = 순매수인지 확인)
        verification_errors = []
        
        # 개인 순매수 검증
        if '개인_매수수량' in df_renamed.columns and '개인_매도수량' in df_renamed.columns and '개인_순매수수량' in df_renamed.columns:
            calculated = df_renamed['개인_매수수량'] - df_renamed['개인_매도수량']
            diff = (calculated - df_renamed['개인_순매수수량']).abs()
            if diff.max() > 0.01:  # 0.01 이상 차이나면 경고
                verification_errors.append(f"개인_순매수수량 불일치 (최대 차이: {diff.max():.2f})")
        
        if '개인_매수금액' in df_renamed.columns and '개인_매도금액' in df_renamed.columns and '개인_순매수금액' in df_renamed.columns:
            calculated = df_renamed['개인_매수금액'] - df_renamed['개인_매도금액']
            diff = (calculated - df_renamed['개인_순매수금액']).abs()
            if diff.max() > 0.01:
                verification_errors.append(f"개인_순매수금액 불일치 (최대 차이: {diff.max():.2f})")
        
        # 외국인 순매수 검증
        if '외국인_매수수량' in df_renamed.columns and '외국인_매도수량' in df_renamed.columns and '외국인_순매수수량' in df_renamed.columns:
            calculated = df_renamed['외국인_매수수량'] - df_renamed['외국인_매도수량']
            diff = (calculated - df_renamed['외국인_순매수수량']).abs()
            if diff.max() > 0.01:
                verification_errors.append(f"외국인_순매수수량 불일치 (최대 차이: {diff.max():.2f})")
        
        if '외국인_매수금액' in df_renamed.columns and '외국인_매도금액' in df_renamed.columns and '외국인_순매수금액' in df_renamed.columns:
            calculated = df_renamed['외국인_매수금액'] - df_renamed['외국인_매도금액']
            diff = (calculated - df_renamed['외국인_순매수금액']).abs()
            if diff.max() > 0.01:
                verification_errors.append(f"외국인_순매수금액 불일치 (최대 차이: {diff.max():.2f})")
        
        # 기관계 순매수 검증
        if '기관계_매수수량' in df_renamed.columns and '기관계_매도수량' in df_renamed.columns and '기관계_순매수수량' in df_renamed.columns:
            calculated = df_renamed['기관계_매수수량'] - df_renamed['기관계_매도수량']
            diff = (calculated - df_renamed['기관계_순매수수량']).abs()
            if diff.max() > 0.01:
                verification_errors.append(f"기관계_순매수수량 불일치 (최대 차이: {diff.max():.2f})")
        
        if '기관계_매수금액' in df_renamed.columns and '기관계_매도금액' in df_renamed.columns and '기관계_순매수금액' in df_renamed.columns:
            calculated = df_renamed['기관계_매수금액'] - df_renamed['기관계_매도금액']
            diff = (calculated - df_renamed['기관계_순매수금액']).abs()
            if diff.max() > 0.01:
                verification_errors.append(f"기관계_순매수금액 불일치 (최대 차이: {diff.max():.2f})")
        
        if verification_errors:
            print("⚠️ 순매수 검증 경고:")
            for error in verification_errors:
                print(f"   - {error}")
        else:
            print("✅ 순매수 검증 통과 (매수 - 매도 = 순매수)")
            
    except Exception as e:
        print(f"❌ 컬럼명 변경 오류: {e}")
        return False

    # 4. 저장
    try:
        # Path 객체를 문자열로 변환
        output_file_str = str(output_file)
        df_renamed.to_csv(output_file_str, index=False, encoding='utf-8-sig')
        print(f"💾 저장 완료: {output_file_str}")
        print(f"   데이터 형태: {df_renamed.shape[0]}행 × {df_renamed.shape[1]}열\n")
        return True
    except Exception as e:
        print(f"❌ 저장 오류: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="주식 데이터 컬럼명 변경 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 단일 파일 처리
  python s02_rename.py --input data/005930_20260101_20260127.csv
  
  # 출력 파일 지정
  python s02_rename.py --input data/005930.csv --output data/005930_renamed.csv
  
  # 여러 파일 처리 (와일드카드)
  python s02_rename.py --input "data/*.csv"
  
  # 여러 파일 처리 (여러 인자)
  python s02_rename.py --input file1.csv file2.csv file3.csv
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        help="입력 파일 경로 (여러 파일 또는 와일드카드 패턴 가능)"
    )
    parser.add_argument(
        "--output", "-o",
        help="출력 파일 경로 (단일 파일 처리시만 사용, 여러 파일 처리시 자동 생성)"
    )
    
    args = parser.parse_args()
    
    # 입력 파일 목록 확장 (와일드카드 처리)
    input_files = []
    for pattern in args.input:
        # 와일드카드가 포함되어 있으면 glob으로 확장
        if '*' in pattern or '?' in pattern:
            matched = glob.glob(pattern)
            if matched:
                input_files.extend(matched)
            else:
                print(f"⚠️ 패턴에 매칭되는 파일이 없습니다: {pattern}")
        else:
            # 일반 파일 경로
            if os.path.exists(pattern):
                input_files.append(pattern)
            else:
                print(f"⚠️ 파일을 찾을 수 없습니다: {pattern}")
    
    if not input_files:
        print("❌ 처리할 파일이 없습니다.")
        return
    
    # 중복 제거 및 정렬
    input_files = sorted(set(input_files))
    
    print(f"🚀 총 {len(input_files)}개 파일 처리 시작...\n")
    
    # 단일 파일 처리시 출력 파일 지정 가능
    if len(input_files) == 1 and args.output:
        success = rename_file(input_files[0], args.output)
    else:
        # 여러 파일 처리시 각각 자동으로 출력 파일명 생성
        success_count = 0
        for input_file in input_files:
            if rename_file(input_file):
                success_count += 1
        
        print(f"\n✨ 처리 완료: {success_count}/{len(input_files)}개 파일 성공")

if __name__ == "__main__":
    main()