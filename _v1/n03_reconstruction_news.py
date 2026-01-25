import pandas as pd
import torch
import glob
import os
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# 스타일 관련 경고 무시 (콘솔을 깨끗하게 유지합니다)
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# 1. 모델 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "clare25/krfinbert-jongtobang"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
model.eval()

# 2. 정제된 2,194개 키워드 로드
with open("d:/stock/news_refiner/refined_keywords.txt", "r", encoding="utf-8") as f:
    keywords = [line.strip() for line in f.readlines()]
keyword_pattern = "|".join(keywords)

def refine_news_to_csv(input_folder, output_csv):
    # .xlsx 및 .xls 파일만 검색
    all_files = glob.glob(os.path.join(input_folder, "*.xlsx")) + glob.glob(os.path.join(input_folder, "*.xls"))
    daily_results = []
    
    print(f"📂 총 {len(all_files)}개의 파일을 검사합니다.")

    for f in all_files:
        # 엑셀 임시 파일(~$로 시작)은 건너뜁니다
        if os.path.basename(f).startswith("~$"):
            continue
            
        try:
            # engine='openpyxl'을 명시하여 에러 방지
            df = pd.read_excel(f, engine='openpyxl')
            
            # 삼성 관련 키워드 필터링
            df = df[df['제목'].str.contains(keyword_pattern, na=False)].copy()
            if df.empty:
                continue

            # 날짜 처리 및 주말 보정
            df['일자'] = pd.to_datetime(df['일자'], format='%Y%m%d', errors='coerce')
            df = df.dropna(subset=['일자'])
            df.loc[df['일자'].dt.dayofweek == 5, '일자'] += pd.Timedelta(days=2)
            df.loc[df['일자'].dt.dayofweek == 6, '일자'] += pd.Timedelta(days=1)

            # 배치 감성 분석
            titles = df['제목'].tolist()
            scores = []
            batch_size = 32
            
            with torch.no_grad():
                for i in range(0, len(titles), batch_size):
                    batch = titles[i : i + batch_size]
                    inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    # 재현님 교정 포인트: 0번(Pos) - 1번(Neg)
                    batch_scores = (probs[:, 0] - probs[:, 1]).cpu().numpy()
                    scores.extend(batch_scores)

            df['sentiment_score'] = scores
            daily_avg = df.groupby('일자')['sentiment_score'].mean().reset_index()
            daily_results.append(daily_avg)
            
            print(f"✅ 처리 완료: {os.path.basename(f)}")

        except Exception as e:
            print(f"⚠️ 건너뜀 ({os.path.basename(f)}): {e}")

    # 최종 병합 및 저장
    if daily_results:
        final_df = pd.concat(daily_results, ignore_index=True)
        final_daily = final_df.groupby('일자')['sentiment_score'].mean().reset_index()
        final_daily.columns = ['날짜', 'news_sentiment']
        final_daily.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n✨ 리파이닝 완료! 결과 저장: {output_csv}")
    else:
        print("❌ 처리된 데이터가 없습니다.")

# 실행 경로 확인하세요
refine_news_to_csv("D:/stock/_data/news/", "D:/stock/_data/refined_news/daily_sentiment_score.csv")