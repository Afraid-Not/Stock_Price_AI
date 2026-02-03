# -*- coding: utf-8 -*-
"""
Transfer Learning 배치 실행
여러 seed로 학습하고 결과 저장
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse
import sys
sys.stdout.reconfigure(encoding='utf-8')

from train_transfer import run_training


def main():
    parser = argparse.ArgumentParser(description='Transfer Learning Batch')
    parser.add_argument('--start', type=int, default=42, help='Start seed')
    parser.add_argument('--end', type=int, default=52, help='End seed')
    parser.add_argument('--output', type=str, default='models_transfer', help='Output dir')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "all_results.csv"
    
    print("="*60)
    print(f"[BATCH] Transfer Learning")
    print(f"   Seeds: {args.start} ~ {args.end}")
    print(f"   Total: {args.end - args.start} runs")
    print("="*60)
    
    all_results = []
    
    # 기존 결과 로드
    if results_file.exists():
        existing = pd.read_csv(results_file)
        all_results = existing.to_dict('records')
        done_seeds = set(existing['seed'].tolist())
        print(f"   Existing results: {len(done_seeds)} seeds")
    else:
        done_seeds = set()
    
    for seed in range(args.start, args.end):
        if seed in done_seeds:
            print(f"\n[{seed}] Already done, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"[SEED {seed}] Starting...")
        print("="*60)
        
        try:
            results, _, _ = run_training(seed=seed, verbose=True)
            all_results.append(results)
            
            # 즉시 저장
            pd.DataFrame(all_results).to_csv(results_file, index=False)
            
            print(f"\n[SEED {seed}] Done!")
            print(f"   Base AUC: {results['base_auc_mean']:.4f}")
            print(f"   Fine AUC: {results['fine_auc_mean']:.4f}")
            print(f"   Improvement: {results['improvement']*100:+.2f}%")
            
        except Exception as e:
            print(f"\n[SEED {seed}] Error: {e}")
            continue
    
    # 최종 결과
    print("\n" + "="*60)
    print("[FINAL RESULTS]")
    print("="*60)
    
    if all_results:
        df = pd.DataFrame(all_results)
        print(f"\nTotal runs: {len(df)}")
        print(f"\nBase Model:")
        print(f"   AUC: {df['base_auc_mean'].mean():.4f} +/- {df['base_auc_mean'].std():.4f}")
        print(f"   F1:  {df['base_f1_mean'].mean():.4f} +/- {df['base_f1_mean'].std():.4f}")
        print(f"\nFine-tuned Model:")
        print(f"   AUC: {df['fine_auc_mean'].mean():.4f} +/- {df['fine_auc_mean'].std():.4f}")
        print(f"   F1:  {df['fine_f1_mean'].mean():.4f} +/- {df['fine_f1_mean'].std():.4f}")
        print(f"\nAvg Improvement: {df['improvement'].mean()*100:+.2f}%")
        
        # Best seed
        best = df.loc[df['fine_auc_mean'].idxmax()]
        print(f"\nBest Seed: {int(best['seed'])}")
        print(f"   Fine AUC: {best['fine_auc_mean']:.4f}")
        print(f"   Fine F1:  {best['fine_f1_mean']:.4f}")
    
    print("\n" + "="*60)
    print("[BATCH DONE]")
    print("="*60)


if __name__ == "__main__":
    main()

