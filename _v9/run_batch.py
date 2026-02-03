"""
CatBoost 배치 실행
- seed를 바꿔가며 자동으로 Optuna + Train 반복
- 결과를 summary.csv에 기록
"""
import subprocess
import sys
import os
import pandas as pd
from datetime import datetime
import time

def run_single(seed, n_trials=50, metric='f1'):
    """단일 seed로 실행"""
    print("\n" + "=" * 70)
    print(f"🚀 SEED {seed} 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    cmd = [
        sys.executable, 'train_cat.py',
        '--mode', 'all',
        '--seed', str(seed),
        '--n_trials', str(n_trials),
        '--metric', metric
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        elapsed = time.time() - start_time
        
        print(f"\n⏱️ SEED {seed} 완료 - 소요시간: {elapsed/60:.1f}분")
        return True, elapsed
        
    except Exception as e:
        print(f"\n❌ SEED {seed} 실패: {e}")
        return False, 0


def run_batch(start_seed=42, end_seed=52, n_trials=50, metric='f1'):
    """배치 실행"""
    print("=" * 70)
    print("🔄 CatBoost 배치 실행 시작")
    print("=" * 70)
    print(f"   Seed 범위: {start_seed} ~ {end_seed-1}")
    print(f"   총 실행 횟수: {end_seed - start_seed}회")
    print(f"   Optuna 시행 횟수: {n_trials}")
    print(f"   평가 지표: {metric}")
    print(f"   시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    results = []
    total_start = time.time()
    
    for seed in range(start_seed, end_seed):
        success, elapsed = run_single(seed, n_trials, metric)
        
        results.append({
            'seed': seed,
            'success': success,
            'elapsed_min': elapsed / 60,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
        # 중간 저장
        df = pd.DataFrame(results)
        df.to_csv('batch_results.csv', index=False)
        
        remaining = end_seed - seed - 1
        if remaining > 0 and elapsed > 0:
            est_remaining = remaining * elapsed / 60
            print(f"\n📊 진행: {seed - start_seed + 1}/{end_seed - start_seed}")
            print(f"   남은 예상 시간: {est_remaining:.0f}분")
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("🎉 배치 실행 완료!")
    print("=" * 70)
    print(f"   총 실행: {len(results)}회")
    print(f"   성공: {sum(r['success'] for r in results)}회")
    print(f"   실패: {sum(not r['success'] for r in results)}회")
    print(f"   총 소요시간: {total_elapsed/60:.1f}분")
    print(f"   결과 저장: batch_results.csv")
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CatBoost 배치 실행")
    parser.add_argument("--start", type=int, default=42, help="시작 seed")
    parser.add_argument("--end", type=int, default=52, help="종료 seed (미포함)")
    parser.add_argument("--n_trials", type=int, default=50, help="Optuna 시행 횟수")
    parser.add_argument("--metric", type=str, default='f1', choices=['f1', 'auc'])
    args = parser.parse_args()
    
    run_batch(
        start_seed=args.start,
        end_seed=args.end,
        n_trials=args.n_trials,
        metric=args.metric
    )


if __name__ == "__main__":
    main()


