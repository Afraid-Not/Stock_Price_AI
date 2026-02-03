import pandas as pd

# 데이터 로드
df = pd.read_csv('_data/merged_with_macro.csv')

# 제외할 컬럼
exclude = ['날짜', 'target', 'stock_code', 'stock_code_encoded',
           '시가', '고가', '저가', '종가', '거래량', '거래대금',
           'stock_name', 'next_rtn']

# 피처 컬럼
features = [c for c in df.columns if c not in exclude]
print(f'피처 수: {len(features)}')
print()
for i, f in enumerate(features):
    print(f'f{i:2d}: {f}')

