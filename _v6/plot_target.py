import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib
from pathlib import Path
import re
from datetime import datetime, timedelta

# 영어 폰트 설정
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def extract_dates_from_filename(filename):
    """파일명에서 날짜 범위 추출"""
    # 파일명 형식: preprocessed_{종목코드}_{시작날짜}_{종료날짜}.csv
    match = re.search(r'(\d{8})_(\d{8})', filename)
    if match:
        start_date_str = match.group(1)
        end_date_str = match.group(2)
        return start_date_str, end_date_str
    return None, None

def create_date_range(start_date_str, end_date_str, n_rows):
    """날짜 범위 생성 (주말 제외)"""
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')
    
    # 주말을 제외한 영업일만 생성
    dates = []
    current_date = start_date
    while current_date <= end_date and len(dates) < n_rows:
        # 월요일(0) ~ 금요일(4)만 포함
        if current_date.weekday() < 5:
            dates.append(current_date)
        current_date += timedelta(days=1)
    
    # 데이터 행 수에 맞춰 조정
    if len(dates) > n_rows:
        dates = dates[:n_rows]
    elif len(dates) < n_rows:
        # 부족한 경우 마지막 날짜부터 추가
        last_date = dates[-1] if dates else start_date
        while len(dates) < n_rows:
            last_date += timedelta(days=1)
            if last_date.weekday() < 5:
                dates.append(last_date)
    
    return dates

def plot_target_for_file(file_path, output_dir=None, scaler_dir="D:/stock/_v6/scalers"):
    """단일 파일의 target을 그래프로 그리기"""
    file_path = Path(file_path)
    
    # 데이터 로드
    df = pd.read_csv(file_path)
    
    if 'target' not in df.columns:
        print(f"⚠️ '{file_path.name}'에 'target' 컬럼이 없습니다.")
        return
    
    # 종목코드 추출
    stock_code_match = re.search(r'preprocessed_(\d{6})', file_path.stem)
    stock_code = stock_code_match.group(1) if stock_code_match else "DEFAULT"
    
    # 스케일러 로드 및 역변환
    scaler_path = Path(scaler_dir) / f"{stock_code}_target_scaler.bin"
    target_values_scaled = df['target'].values.reshape(-1, 1)
    
    if scaler_path.exists():
        try:
            target_scaler = joblib.load(scaler_path)
            target_values = target_scaler.inverse_transform(target_values_scaled).flatten()
            print(f"   ✅ Scaler loaded and inverse transformed: {scaler_path}")
        except Exception as e:
            print(f"   ⚠️ Failed to load scaler: {e}")
            print(f"   Using scaled values (range: -0.3 ~ 0.3)")
            target_values = target_values_scaled.flatten()
    else:
        # DEFAULT 스케일러 시도
        default_scaler_path = Path(scaler_dir) / "DEFAULT_target_scaler.bin"
        if default_scaler_path.exists():
            try:
                target_scaler = joblib.load(default_scaler_path)
                target_values = target_scaler.inverse_transform(target_values_scaled).flatten()
                print(f"   ⚠️ Using DEFAULT scaler: {default_scaler_path}")
            except Exception as e:
                print(f"   ⚠️ Failed to load DEFAULT scaler: {e}")
                print(f"   Using scaled values (range: -0.3 ~ 0.3)")
                target_values = target_values_scaled.flatten()
        else:
            print(f"   ⚠️ Scaler not found: {scaler_path}")
            print(f"   Using scaled values (range: -0.3 ~ 0.3)")
            target_values = target_values_scaled.flatten()
    
    # 파일명에서 날짜 범위 추출
    start_date_str, end_date_str = extract_dates_from_filename(file_path.stem)
    
    if start_date_str and end_date_str:
        # 날짜 범위 생성
        dates = create_date_range(start_date_str, end_date_str, len(df))
    else:
        # 날짜 정보가 없으면 인덱스 사용
        print(f"   ⚠️ '{file_path.name}'에서 날짜 정보를 추출할 수 없습니다. 인덱스를 사용합니다.")
        dates = range(len(df))
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # target을 백분율로 변환
    target_values_pct = target_values * 100  # 백분율로 변환
    
    ax.plot(dates, target_values_pct, linewidth=1.5, alpha=0.7, color='blue', label='Return Rate')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero Line')
    
    # 양수/음수 영역 색칠
    ax.fill_between(dates, 0, target_values_pct, where=(target_values_pct >= 0), 
                     alpha=0.3, color='green', label='Positive Returns')
    ax.fill_between(dates, 0, target_values_pct, where=(target_values_pct < 0), 
                     alpha=0.3, color='red', label='Negative Returns')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Return Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Stock Return Rate Over Time (Stock Code: {stock_code})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # x축 날짜 포맷
    if isinstance(dates[0], datetime):
        fig.autofmt_xdate()
    
    plt.tight_layout()
    
    # 저장
    if output_dir is None:
        output_dir = file_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{stock_code}_target_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graph saved: {output_path}")
    plt.close()

def plot_all_targets(data_dir="D:/stock/_v6/_data", output_dir=None, scaler_dir="D:/stock/_v6/scalers"):
    """데이터 디렉토리의 모든 파일에 대해 target 그래프 생성"""
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"❌ 디렉토리가 없습니다: {data_dir}")
        return
    
    # CSV 파일 찾기
    csv_files = list(data_dir.glob("preprocessed_*.csv"))
    
    if not csv_files:
        print(f"⚠️ '{data_dir}'에 전처리된 CSV 파일이 없습니다.")
        return
    
    print(f"📊 Found {len(csv_files)} files")
    print(f"{'='*60}")
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = data_dir / "target_plots"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 각 파일에 대해 그래프 생성
    for i, file_path in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Processing: {file_path.name}")
        try:
            plot_target_for_file(file_path, output_dir, scaler_dir)
        except Exception as e:
            print(f"   ❌ Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"✅ All graphs saved to: {output_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot target (return rate) from preprocessed data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 모든 파일에 대해 그래프 생성
  python plot_target.py --data_dir D:/stock/_v6/_data
  
  # 특정 파일만 그래프 생성
  python plot_target.py --file D:/stock/_v6/_data/preprocessed_005930_20081001_20260127.csv
  
  # 출력 디렉토리 지정
  python plot_target.py --data_dir D:/stock/_v6/_data --output_dir D:/stock/_v6/plots
        """
    )
    
    parser.add_argument("--file", type=str, default=None, help="특정 파일 경로 (단일 파일만 처리)")
    parser.add_argument("--data_dir", type=str, default="D:/stock/_v6/_data", help="데이터 디렉토리")
    parser.add_argument("--output_dir", type=str, default=None, help="그래프 저장 디렉토리 (기본값: data_dir/target_plots)")
    parser.add_argument("--scaler_dir", type=str, default="D:/stock/_v6/scalers", help="스케일러 디렉토리")
    
    args = parser.parse_args()
    
    if args.file:
        # 단일 파일 처리
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ 파일이 없습니다: {file_path}")
            return
        
        output_dir = args.output_dir if args.output_dir else file_path.parent
        plot_target_for_file(file_path, output_dir, args.scaler_dir)
    else:
        # 모든 파일 처리
        plot_all_targets(args.data_dir, args.output_dir, args.scaler_dir)

if __name__ == "__main__":
    main()

