# -*- coding: utf-8 -*-
"""
kr-finbert를 사용한 뉴스 감성 분석
LLM 분석 결과와 비교용
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class FinBertAnalyzer:
    def __init__(self, model_name: str = "snunlp/KR-FinBert-SC"):
        """
        kr-finbert 모델 로드
        snunlp/KR-FinBert-SC: 한국어 금융 감성 분석 모델
        """
        print(f"📦 모델 로딩 중: {model_name}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # 레이블 매핑 (KR-FinBert-SC는 negative, neutral, positive)
        self.label_map = {0: -1.0, 1: 0.0, 2: 1.0}  # negative, neutral, positive
        print("✅ 모델 로딩 완료!")
    
    def analyze_text(self, text: str) -> dict:
        """단일 텍스트 감성 분석"""
        if not text or pd.isna(text):
            return {'sentiment': 0.0, 'confidence': 0.0, 'label': 'neutral'}
        
        # 텍스트 전처리 (너무 긴 텍스트 자르기)
        text = str(text)[:512]
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_label = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_label].item()
            
            # 감성 점수 계산 (확률 가중 평균)
            sentiment_score = (
                probs[0][0].item() * (-1.0) +  # negative
                probs[0][1].item() * 0.0 +     # neutral
                probs[0][2].item() * 1.0       # positive
            )
            
            labels = ['negative', 'neutral', 'positive']
            return {
                'sentiment': sentiment_score,
                'confidence': confidence,
                'label': labels[pred_label]
            }
        except Exception as e:
            return {'sentiment': 0.0, 'confidence': 0.0, 'label': 'neutral'}
    
    def analyze_batch(self, texts: list, batch_size: int = 32) -> list:
        """배치 처리로 빠른 분석"""
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="분석 중"):
            batch_texts = texts[i:i+batch_size]
            # None이나 빈 문자열 처리
            batch_texts = [str(t)[:512] if t and not pd.isna(t) else "" for t in batch_texts]
            
            try:
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                
                for j in range(len(batch_texts)):
                    sentiment_score = (
                        probs[j][0].item() * (-1.0) +
                        probs[j][1].item() * 0.0 +
                        probs[j][2].item() * 1.0
                    )
                    pred_label = torch.argmax(probs[j]).item()
                    labels = ['negative', 'neutral', 'positive']
                    
                    results.append({
                        'sentiment': sentiment_score,
                        'confidence': probs[j][pred_label].item(),
                        'label': labels[pred_label]
                    })
            except Exception as e:
                # 에러 시 neutral로 처리
                for _ in batch_texts:
                    results.append({'sentiment': 0.0, 'confidence': 0.0, 'label': 'neutral'})
        
        return results


def analyze_news_file(news_path: str, output_dir: str = "_data/news_sentiment_finbert"):
    """뉴스 파일을 kr-finbert로 분석"""
    
    # 파일 로드
    print(f"\n📂 뉴스 파일 로딩: {news_path}")
    df = pd.read_csv(news_path, encoding='utf-8')
    print(f"   총 뉴스: {len(df):,}건")
    
    stock_code = Path(news_path).stem.split('_')[1]
    
    # FinBert 분석기 초기화
    analyzer = FinBertAnalyzer()
    
    # 뉴스 제목 추출
    titles = df['HTS_공시_제목_내용'].tolist()
    
    # 배치 분석
    print("\n🔍 감성 분석 시작...")
    results = analyzer.analyze_batch(titles, batch_size=32)
    
    # 결과 추가
    df['finbert_sentiment'] = [r['sentiment'] for r in results]
    df['finbert_confidence'] = [r['confidence'] for r in results]
    df['finbert_label'] = [r['label'] for r in results]
    
    # 날짜 컬럼 생성
    df['날짜'] = pd.to_datetime(df['작성일자'], format='%Y%m%d')
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 개별 결과 저장
    sentiment_file = output_path / f"finbert_{stock_code}.csv"
    df.to_csv(sentiment_file, index=False, encoding='utf-8-sig')
    print(f"💾 개별 분석 저장: {sentiment_file}")
    
    # 일별 집계
    daily_df = df.groupby('날짜').agg({
        'finbert_sentiment': 'mean',
        'finbert_confidence': 'mean',
        'finbert_label': lambda x: x.value_counts().index[0] if len(x) > 0 else 'neutral',  # 최빈값
        'stock_code': 'first'
    }).reset_index()
    daily_df['news_count'] = df.groupby('날짜').size().values
    daily_df = daily_df.rename(columns={'finbert_sentiment': 'news_sentiment'})
    
    daily_file = output_path / f"daily_finbert_{stock_code}.csv"
    daily_df.to_csv(daily_file, index=False, encoding='utf-8-sig')
    print(f"💾 일별 집계 저장: {daily_file}")
    
    return df, daily_df


def compare_with_llm(stock_code: str = "000660"):
    """LLM 결과와 FinBert 결과 비교"""
    
    llm_path = f"_data/news_sentiment/daily_{stock_code}.csv"
    finbert_path = f"_data/news_sentiment_finbert/daily_finbert_{stock_code}.csv"
    
    print("\n" + "="*60)
    print("📊 LLM vs FinBert 비교 분석")
    print("="*60)
    
    # LLM 결과 로드
    llm_df = pd.read_csv(llm_path)
    llm_df['날짜'] = pd.to_datetime(llm_df['날짜'])
    
    # FinBert 결과 로드
    finbert_df = pd.read_csv(finbert_path)
    finbert_df['날짜'] = pd.to_datetime(finbert_df['날짜'])
    
    # 병합
    merged = pd.merge(
        llm_df[['날짜', 'news_sentiment', 'news_count']],
        finbert_df[['날짜', 'news_sentiment', 'finbert_confidence']],
        on='날짜',
        suffixes=('_llm', '_finbert')
    )
    
    # 상관관계 분석
    correlation = merged['news_sentiment_llm'].corr(merged['news_sentiment_finbert'])
    
    print(f"\n📈 분석 결과:")
    print(f"   - 비교 일수: {len(merged)}일")
    print(f"   - 상관계수: {correlation:.4f}")
    
    # 기본 통계
    print(f"\n📊 LLM 감성 통계:")
    print(f"   - 평균: {merged['news_sentiment_llm'].mean():.4f}")
    print(f"   - 표준편차: {merged['news_sentiment_llm'].std():.4f}")
    print(f"   - 최소/최대: {merged['news_sentiment_llm'].min():.4f} / {merged['news_sentiment_llm'].max():.4f}")
    
    print(f"\n📊 FinBert 감성 통계:")
    print(f"   - 평균: {merged['news_sentiment_finbert'].mean():.4f}")
    print(f"   - 표준편차: {merged['news_sentiment_finbert'].std():.4f}")
    print(f"   - 최소/최대: {merged['news_sentiment_finbert'].min():.4f} / {merged['news_sentiment_finbert'].max():.4f}")
    
    # 방향 일치율 (부호가 같은 비율)
    same_direction = ((merged['news_sentiment_llm'] * merged['news_sentiment_finbert']) >= 0).sum()
    direction_rate = same_direction / len(merged) * 100
    print(f"\n🎯 방향 일치율: {direction_rate:.1f}% ({same_direction}/{len(merged)})")
    
    # 비교 결과 저장
    comparison_file = f"_data/news_sentiment_finbert/comparison_{stock_code}.csv"
    merged.to_csv(comparison_file, index=False, encoding='utf-8-sig')
    print(f"\n💾 비교 결과 저장: {comparison_file}")
    
    # 시각적으로 몇 개 샘플 비교
    print("\n" + "="*60)
    print("📝 샘플 비교 (감성 차이가 큰 날)")
    print("="*60)
    merged['diff'] = abs(merged['news_sentiment_llm'] - merged['news_sentiment_finbert'])
    top_diff = merged.nlargest(5, 'diff')
    
    for _, row in top_diff.iterrows():
        print(f"\n날짜: {row['날짜'].strftime('%Y-%m-%d')}")
        print(f"   LLM: {row['news_sentiment_llm']:+.3f}")
        print(f"   FinBert: {row['news_sentiment_finbert']:+.3f}")
        print(f"   차이: {row['diff']:.3f}")
    
    return merged, correlation


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='kr-finbert 뉴스 감성 분석')
    parser.add_argument('--stock', type=str, default='000660', help='종목코드')
    parser.add_argument('--compare', action='store_true', help='LLM과 비교')
    args = parser.parse_args()
    
    news_file = f"_data/news/news_{args.stock}_20250203_20260203.csv"
    
    # FinBert 분석 실행
    df, daily_df = analyze_news_file(news_file)
    
    # LLM과 비교
    if args.compare:
        compare_with_llm(args.stock)
    
    print("\n✅ 완료!")

