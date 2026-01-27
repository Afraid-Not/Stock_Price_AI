import pandas as pd
import numpy as np
import os
import torch
import glob
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

class NewsSentimentAnalyzer:
    def __init__(self, keywords_path, model_name="snunlp/KR-FinBert-SC"):
        # 1. 키워드 로드
        if not os.path.exists(keywords_path):
            raise FileNotFoundError(f"키워드 파일을 찾을 수 없습니다: {keywords_path}")
            
        with open(keywords_path, 'r', encoding='utf-8') as f:
            # 빈 줄 제외하고 리스트업
            self.keywords = [line.strip() for line in f if line.strip()]
        
        # 키워드 필터링용 정규표현식 (패턴이 너무 길면 에러날 수 있어 유의해야 함)
        self.keyword_pattern = '|'.join([re.escape(k) for k in self.keywords])
        
        # 2. 모델 및 토크나이저 로드
        print(f"📦 모델 로드 중: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def filter_relevant_news(self, df):
        """키워드 열에 관련 단어가 포함된 뉴스만 필터링"""
        if '키워드' not in df.columns:
            print("⚠️ '키워드' 컬럼이 데이터에 없습니다.")
            return pd.DataFrame()
        
        # [수정] engine 인자 삭제 및 regex=True 명시
        # na=False로 결측치는 무시하고 패턴 포함 여부 확인
        mask = df['키워드'].str.contains(self.keyword_pattern, na=False, regex=True)
        return df[mask].copy()

    def get_sentiment_scores(self, titles, batch_size=64):
        """제목 리스트에 대해 감성 점수 산출 (-1 ~ 1)"""
        all_scores = []
        
        # tqdm으로 진행 상황 표시
        for i in tqdm(range(0, len(titles), batch_size), desc="감성 분석 중"):
            batch_titles = titles[i:i+batch_size]
            inputs = self.tokenizer(batch_titles, return_tensors="pt", padding=True, 
                                    truncation=True, max_length=128).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
                
                # 점수 계산: (긍정 확률 * 1) + (부정 확률 * -1)
                # 라벨: 0(부정), 1(중립), 2(긍정)
                scores = (probs[:, 2] * 1) + (probs[:, 0] * -1)
                all_scores.extend(scores)
        
        return all_scores

def run_sentiment_pipeline():
    # 경로 설정 (재현님 환경에 맞게 확인 필요)
    news_dir = "D:/stock/_data/news"
    keywords_file = "D:/stock/_v3/_data/refined_keywords.txt" # 이 파일이 실행 스크립트와 같은 경로에 있어야 함
    output_path = "D:/stock/_v3/_data/daily_news_sentiment.csv"
    
    analyzer = NewsSentimentAnalyzer(keywords_file)
    excel_files = glob.glob(os.path.join(news_dir, "*.xlsx"))
    
    if not excel_files:
        print(f"❌ '{news_dir}' 경로에 엑셀 파일이 없습니다.")
        return

    daily_results = []

    for file in excel_files:
        print(f"\n📖 파일 읽기: {os.path.basename(file)}")
        # 필요한 열만 로드하여 메모리 절약
        try:
            df = pd.read_excel(file, usecols=['일자', '제목', '키워드'])
        except Exception as e:
            print(f"⚠️ 파일 로드 실패 ({file}): {e}")
            continue
        
        # 1. 키워드 필터링
        filtered_df = analyzer.filter_relevant_news(df)
        if filtered_df.empty:
            print("⏩ 매칭되는 키워드 뉴스가 없어 건너뜁니다.")
            continue
            
        print(f"✅ 필터링 완료: {len(df)}건 중 {len(filtered_df)}건 선별")

        # 2. 감성 점수 계산
        titles = filtered_df['제목'].astype(str).tolist()
        filtered_df['sentiment_score'] = analyzer.get_sentiment_scores(titles)

        # 3. 날짜 형식 정리 (YYYYMMDD)
        filtered_df['일자'] = pd.to_datetime(filtered_df['일자'].astype(str)).dt.strftime('%Y%m%d')
        
        # 4. 일자별 평균 점수 집계
        daily_avg = filtered_df.groupby('일자')['sentiment_score'].mean().reset_index()
        daily_results.append(daily_avg)

    # 전체 결과 합산
    if daily_results:
        final_df = pd.concat(daily_results, ignore_index=True)
        # 같은 날짜가 여러 파일에 걸쳐 있을 수 있으므로 다시 한번 평균
        final_daily_sentiment = final_df.groupby('일자')['sentiment_score'].mean().reset_index()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_daily_sentiment.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✨ 최종 일별 감성 점수 저장 완료: {output_path}")
    else:
        print("❌ 분석된 데이터가 없습니다.")

if __name__ == "__main__":
    run_sentiment_pipeline()