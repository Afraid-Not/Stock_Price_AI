"""데이터 누수 체크"""
import pandas as pd
import numpy as np

df = pd.read_csv('_data/merged_with_macro.csv')

print("=" * 60)
print("🔍 데이터 누수 체크")
print("=" * 60)

# 1. 코스피 vs 내일 수익률
print("\n1️⃣ 코스피 당일 등락률 vs 내일 수익률 (target)")
if 'kospi_return' in df.columns and 'next_rtn' in df.columns:
    corr = df['kospi_return'].corr(df['next_rtn'])
    print(f"   상관계수: {corr:.4f}")
    if abs(corr) > 0.3:
        print("   ⚠️ 높은 상관관계 - 누수 가능성!")
    else:
        print("   ✅ 낮은 상관관계 - 정상")

# 2. 환율 vs 내일 수익률
print("\n2️⃣ 환율 당일 변화율 vs 내일 수익률")
if 'usdkrw_return' in df.columns and 'next_rtn' in df.columns:
    corr = df['usdkrw_return'].corr(df['next_rtn'])
    print(f"   상관계수: {corr:.4f}")
    if abs(corr) > 0.3:
        print("   ⚠️ 높은 상관관계 - 누수 가능성!")
    else:
        print("   ✅ 낮은 상관관계 - 정상")

# 3. 안전하게: 전일 매크로 사용 시뮬레이션
print("\n3️⃣ 전일 코스피 vs 내일 수익률 (Lag1)")
df['kospi_return_lag1'] = df['kospi_return'].shift(1)
if 'kospi_return_lag1' in df.columns and 'next_rtn' in df.columns:
    corr = df['kospi_return_lag1'].corr(df['next_rtn'])
    print(f"   상관계수: {corr:.4f}")

# 4. 피처별 타겟 상관관계
print("\n4️⃣ 모든 피처 vs 내일 수익률 상관관계 (상위 10개)")
exclude = ['날짜', 'target', 'stock_code', 'stock_name', 'next_rtn']
features = [c for c in df.columns if c not in exclude]

correlations = []
for f in features:
    if df[f].dtype in ['float64', 'int64', 'float32', 'int32']:
        corr = df[f].corr(df['next_rtn'])
        correlations.append((f, corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)
print("-" * 40)
for f, c in correlations[:10]:
    flag = "⚠️" if abs(c) > 0.1 else "  "
    print(f"   {flag} {f:25s}: {c:+.4f}")

print("\n" + "=" * 60)
print("💡 결론")
print("=" * 60)
print("""
- 상관계수 0.1 이상: 예측에 도움되는 피처
- 상관계수 0.3 이상: 누수 의심 (너무 높음)
- 당일 매크로가 내일 수익률과 높은 상관관계면:
  → 전일(Lag1) 매크로로 대체 권장
""")

