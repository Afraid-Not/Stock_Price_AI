import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# 1. 데이터 로드 (재현님의 파일 경로)
file_path = "D:/stock/_data/news/NewsResult_20251201-20251231.xlsx"
df = pd.read_excel(file_path, index_col=0)

# 2. 모든 키워드 추출 및 빈도 계산
all_keywords = []
df['키워드'].dropna().apply(lambda x: all_keywords.extend(x.split(',')))

word_counts = Counter(all_keywords)
word_freq_df = pd.DataFrame(word_counts.items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False)

# --- 결과 출력 ---
print("### 1. 전체 데이터셋 상위 20개 키워드 ###")
print(word_freq_df.head(20))
print("\n### 2. 전체 데이터셋 하위 20개 키워드 (빈도 1회) ###")
print(word_freq_df.tail(20))

# 3. 순서 신뢰도 체크 로직
# 각 행의 [첫 번째 키워드]와 [마지막 키워드]의 전체 빈도를 비교해봅니다.
def check_order_consistency(row):
    if pd.isna(row['키워드']): return None
    kws = row['키워드'].split(',')
    if len(kws) < 10: return None # 키워드가 너무 적은 행은 제외
    
    first_val = word_counts.get(kws[0], 0)
    last_val = word_counts.get(kws[-1], 0)
    
    # 첫 번째 단어의 빈도가 마지막 단어보다 크면 True (정렬되어 있을 가능성 높음)
    return first_val > last_val

consistency_results = df.apply(check_order_consistency, axis=1).dropna()
consistency_rate = consistency_results.mean() * 100

print(f"\n### 3. 정렬 신뢰도 분석 결과 ###")
print(f"전체 행 중 '앞 단어 빈도 > 뒷 단어 빈도' 비율: {consistency_rate:.2f}%")
if consistency_rate > 70:
    print("결론: 키워드가 중요도(빈도) 순으로 어느 정도 정렬되어 있습니다.")
else:
    print("결론: 키워드 순서가 중요도와 큰 상관이 없습니다. 별도의 정렬이 필요합니다.")