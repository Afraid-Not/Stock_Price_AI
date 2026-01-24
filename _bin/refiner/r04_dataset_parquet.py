import pandas as pd
import glob
import pickle
import os
from tqdm import tqdm

# 1. 경로 설정
input_path = "D:/stock/_data/news/"
output_path = "D:/stock/_data/parquet/"
counts_path = "D:/stock/_data/pseudo/global_counts.pkl"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# 2. 전역 빈도 사전 로드
with open(counts_path, "rb") as f:
    global_counts = pickle.load(f)

# 3. 빈도 기반 정렬 함수
def sort_by_freq(text, cat_name):
    if pd.isna(text) or text == "":
        return ""
    words = [w.strip() for w in str(text).split(',') if w.strip()]
    # 글로벌 빈도 높은 순 정렬
    sorted_words = sorted(words, key=lambda x: global_counts[cat_name].get(x, 0), reverse=True)
    return ",".join(sorted_words)

# 4. 파일 변환 루프
file_list = glob.glob(os.path.join(input_path, "*.xlsx"))
categories = {
    '인물': 'person',
    '위치': 'place',
    '기관': 'institute',
    '키워드': 'keyword',
    '특성추출(가중치순 상위 50개)': 'features'
}

print(f"🚀 총 {len(file_list)}개 파일 변환 시작 (뉴스 식별자 포함)...")

for file in tqdm(file_list):
    # 0번째 열을 인덱스가 아닌 일반 데이터로 읽기 위해 index_col=None 설정
    df = pd.read_excel(file, engine='openpyxl', index_col=None)
    
    # 0번째 열의 이름이 '뉴스 식별자'가 아닐 경우를 대비해 강제 지정
    if df.columns[0] != '뉴스 식별자':
        df.rename(columns={df.columns[0]: '뉴스 식별자'}, inplace=True)
    
    # 비지도 학습에 필요한 5개 뷰 정렬 적용
    for col, eng_name in categories.items():
        if col in df.columns:
            df[f'sorted_{eng_name}'] = df[col].apply(lambda x: sort_by_freq(x, col))
        else:
            df[f'sorted_{eng_name}'] = "" # 컬럼이 없는 경우 빈 문자열 처리
    
    # 최종적으로 저장할 컬럼들 (식별자 + 분류 + 정렬된 데이터)
    target_cols = [
        '뉴스 식별자', '일자', '제목', '통합 분류1', '사건/사고 분류1',
        'sorted_person', 'sorted_place', 'sorted_institute', 'sorted_keyword', 'sorted_features'
    ]
    
    # 데이터프레임 필터링 (존재하는 컬럼만 선택)
    available_cols = [c for c in target_cols if c in df.columns]
    df_refined = df[available_cols]
    
    # Parquet 저장
    file_name = os.path.basename(file).replace('.xlsx', '.parquet')
    df_refined.to_parquet(os.path.join(output_path, file_name), engine='pyarrow', index=False)

print(f"✨ 모든 파일이 {output_path}에 Parquet 형식으로 저장되었습니다!")