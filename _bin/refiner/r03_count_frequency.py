import pandas as pd
import glob
from collections import Counter
import pickle
import os

# 파일 경로 패턴 (재현님의 폴더 구조에 맞춰 수정하세요)
file_paths = glob.glob("D:/stock/_data/news/NewsResult_*.xlsx")
categories = ['인물', '위치', '기관', '키워드', '특성추출(가중치순 상위 50개)']

# 전역 빈도를 저장할 사전
global_counts = {cat: Counter() for cat in categories}

print("🚀 48개 파일 전역 빈도 집계 시작...")
for path in file_paths:
    df = pd.read_excel(path)
    for cat in categories:
        # 단어 분리 후 빈도 업데이트
        df[cat].dropna().apply(lambda x: global_counts[cat].update([w.strip() for w in str(x).split(',') if w.strip()]))
    print(f"✅ 처리 완료: {os.path.basename(path)}")

# 나중을 위해 빈도 사전 저장
with open("./_data/pseudo/global_counts.pkl", "wb") as f:
    pickle.dump(global_counts, f)