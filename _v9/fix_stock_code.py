"""종목 코드 제로패딩 수정"""
import pandas as pd

# 병합 파일 로드
df = pd.read_csv('_data/merged_all_stocks_20260131.csv')

print('=== 수정 전 ===')
print(f'stock_code dtype: {df["stock_code"].dtype}')
print(f'샘플 값: {list(df["stock_code"].unique()[:10])}')

# 종목 코드 수정: 문자열 변환 + 6자리 제로패딩
df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)

print()
print('=== 수정 후 ===')
print(f'stock_code dtype: {df["stock_code"].dtype}')
print(f'샘플 값: {list(df["stock_code"].unique()[:10])}')

# 저장 (새 파일명으로)
output_path = '_data/merged_all_stocks_20260131_fixed.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print()
print(f'✅ 저장 완료: {output_path}')

