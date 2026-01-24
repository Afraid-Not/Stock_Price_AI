import pandas as pd
import glob
import os
import pickle
from tqdm import tqdm

# 1. 경로 및 설정
parquet_dir = r"D:\stock\_data\parquet"
output_dir = r"D:\stock\_data\pseudo"
counts_path = r"D:\stock\_data\pseudo\global_counts.pkl"
combined_file = os.path.join(output_dir, "news_total_sorted.parquet")

# 2. 전역 빈도 사전 로드
with open(counts_path, "rb") as f:
    global_counts = pickle.load(f)

# 3. 빈도 기반 정렬 함수 정의
def sort_by_global_freq(text, cat_name):
    if pd.isna(text) or text == "":
        return ""
    # 단어 분리
    words = [w.strip() for w in str(text).split(',') if w.strip()]
    # global_counts에 근거하여 내림차순 정렬
    # counts 사전에 없는 단어는 0으로 처리
    sorted_words = sorted(words, key=lambda x: global_counts[cat_name].get(x, 0), reverse=True)
    return ",".join(sorted_words)

# 4. 파일 목록 가져오기
file_list = glob.glob(os.path.join(parquet_dir, "*.parquet"))
file_list.sort()

# 5. 매핑 정보 (원본 컬럼명: 카테고리 키)
categories = {
    'sorted_person': '인물',
    'sorted_place': '위치',
    'sorted_institute': '기관',
    'sorted_keyword': '키워드',
    'sorted_features': '특성추출(가중치순 상위 50개)'
}

# 6. 병합 및 정렬 프로세스
dfs = []
print(f"🚀 {len(file_list)}개 파일 병합 및 전역 빈도 정렬 시작...")

for file in tqdm(file_list, desc="파일 처리 중"):
    temp_df = pd.read_parquet(file)
    
    # 5개 주요 뷰에 대해 전역 빈도 기반 재정렬 수행
    for col, cat_key in categories.items():
        if col in temp_df.columns:
            temp_df[col] = temp_df[col].apply(lambda x: sort_by_global_freq(x, cat_key))
            
    dfs.append(temp_df)

# 7. 최종 병합 및 시계열 정렬
if dfs:
    print("🔄 최종 병합 및 식별자 정렬 중...")
    df_total = pd.concat(dfs, ignore_index=True)
    df_total = df_total.sort_values(by='뉴스 식별자').reset_index(drop=True)
    
    # 최종 저장
    df_total.to_parquet(combined_file, engine='pyarrow', compression='snappy', index=False)
    
    print(f"✨ 완료! 최종 데이터 수: {len(df_total):,} 행")
    print(f"📍 경로: {combined_file}")