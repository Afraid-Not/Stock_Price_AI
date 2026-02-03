"""
뉴스 감성 분석 스크립트
LLM(GPT) 또는 FinBERT를 사용하여 뉴스 감성 점수를 계산합니다.
"""
import pandas as pd
import numpy as np
import os
import sys
import time
import argparse
from datetime import datetime
from dotenv import load_dotenv

# .env 로드
load_dotenv('D:/stock/.env')


class NewsAnalyzer:
    """뉴스 감성 분석 클래스"""
    
    def __init__(self, method: str = "llm", model: str = None):
        """
        Args:
            method: "llm" (GPT) 또는 "finbert" (로컬)
            model: 모델명 (llm일 때만 사용)
        """
        self.method = method
        self.model = model or "gpt-4o-mini"
        self.client = None
        
        if method == "llm":
            self._init_llm()
        elif method == "finbert":
            self._init_finbert()
    
    def _init_llm(self):
        """OpenAI 클라이언트 초기화"""
        try:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
            self.client = OpenAI(api_key=api_key)
            print(f"✅ OpenAI 초기화 완료 (모델: {self.model})")
        except ImportError:
            print("❌ openai 패키지가 없습니다. pip install openai")
            raise
    
    def _init_finbert(self):
        """FinBERT 모델 초기화"""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            model_name = "snunlp/KR-FinBert-SC"
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🔧 FinBERT 로딩 중... (디바이스: {self.device})")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model_bert = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model_bert.to(self.device)
            self.model_bert.eval()
            print("✅ FinBERT 로딩 완료")
        except ImportError:
            print("❌ transformers 패키지가 없습니다. pip install transformers torch")
            raise
    
    def analyze_single_llm(self, title: str, stock_name: str = None) -> dict:
        """LLM으로 단일 뉴스 분석"""
        stock_context = f"'{stock_name}' 종목에 대한 " if stock_name else ""
        
        prompt = f"""다음은 {stock_context}뉴스 제목입니다. 주가에 미칠 영향을 분석해주세요.

뉴스 제목: "{title}"

JSON 형식으로만 응답:
{{"sentiment": 0.0, "impact": 3, "event_type": "일반"}}

- sentiment: -1.0(매우부정) ~ +1.0(매우긍정)
- impact: 1(낮음) ~ 5(높음)
- event_type: 실적/배당/계약/투자/규제/시장/일반"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )
            text = response.choices[0].message.content
            
            # JSON 파싱
            import json
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                result = json.loads(text[start:end])
                return {
                    'sentiment': max(-1, min(1, float(result.get('sentiment', 0)))),
                    'impact': max(1, min(5, int(result.get('impact', 3)))),
                    'event_type': result.get('event_type', '일반')
                }
        except Exception as e:
            pass
        
        return {'sentiment': 0.0, 'impact': 3, 'event_type': '일반'}
    
    def analyze_single_finbert(self, title: str) -> dict:
        """FinBERT로 단일 뉴스 분석"""
        import torch
        
        if not title or pd.isna(title):
            return {'sentiment': 0.0, 'impact': 3, 'event_type': '일반'}
        
        try:
            inputs = self.tokenizer(
                str(title)[:512],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model_bert(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                scores = probs[0].cpu().tolist()
            
            # 감성 점수 계산 (긍정 - 부정)
            sentiment = scores[2] - scores[0]  # 긍정 - 부정
            
            return {
                'sentiment': round(sentiment, 4),
                'impact': 3,  # FinBERT는 영향도 제공 안함
                'event_type': '일반'
            }
        except Exception as e:
            return {'sentiment': 0.0, 'impact': 3, 'event_type': '일반'}
    
    def analyze_dataframe(self, df: pd.DataFrame, title_col: str = "HTS_공시_제목_내용",
                          stock_name_col: str = "stock_name", delay: float = 0.3,
                          batch_size: int = 100) -> pd.DataFrame:
        """DataFrame 전체 분석"""
        
        print(f"\n📊 감성 분석 시작 (방식: {self.method})")
        print(f"   총 {len(df):,}건")
        
        results = []
        total = len(df)
        
        for i, (_, row) in enumerate(df.iterrows()):
            title = row.get(title_col, '')
            stock_name = row.get(stock_name_col, '')
            
            if self.method == "llm":
                result = self.analyze_single_llm(title, stock_name)
                time.sleep(delay)
            else:
                result = self.analyze_single_finbert(title)
            
            results.append(result)
            
            # 진행률 표시
            if (i + 1) % batch_size == 0:
                pct = (i + 1) / total * 100
                print(f"   진행: {i + 1:,}/{total:,} ({pct:.1f}%)")
        
        # 결과 추가
        df_result = df.copy()
        df_result['sentiment'] = [r['sentiment'] for r in results]
        df_result['impact'] = [r['impact'] for r in results]
        df_result['event_type'] = [r['event_type'] for r in results]
        
        # 통계
        print(f"\n✅ 분석 완료!")
        print(f"   평균 감성: {df_result['sentiment'].mean():.4f}")
        print(f"   긍정 뉴스: {len(df_result[df_result['sentiment'] > 0.3]):,}건")
        print(f"   부정 뉴스: {len(df_result[df_result['sentiment'] < -0.3]):,}건")
        
        return df_result
    
    def aggregate_daily(self, df: pd.DataFrame, date_col: str = "작성일자",
                        stock_col: str = "stock_code") -> pd.DataFrame:
        """종목별/날짜별 집계"""
        
        # 날짜 형식 변환
        df['날짜'] = df[date_col].astype(str).apply(
            lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if len(x) == 8 and x.isdigit() else x
        )
        
        # 가중 평균 계산
        df['weighted'] = df['sentiment'] * df['impact']
        
        # 종목별/날짜별 집계
        df_daily = df.groupby([stock_col, '날짜']).agg({
            'weighted': 'sum',
            'impact': 'sum',
            'sentiment': ['mean', 'count']
        }).reset_index()
        
        df_daily.columns = [stock_col, '날짜', 'weighted_sum', 'impact_sum', 
                           'sentiment_mean', 'news_count']
        
        # 가중 평균
        df_daily['news_sentiment'] = (df_daily['weighted_sum'] / df_daily['impact_sum']).round(4)
        df_daily['news_sentiment_simple'] = df_daily['sentiment_mean'].round(4)
        
        # 최종 컬럼 선택
        df_daily = df_daily[[stock_col, '날짜', 'news_sentiment', 'news_sentiment_simple', 'news_count']]
        df_daily = df_daily.sort_values([stock_col, '날짜']).reset_index(drop=True)
        
        print(f"\n📊 일별 집계 완료")
        print(f"   총 {len(df_daily):,}건 (종목×날짜)")
        
        return df_daily


def analyze_all_news(input_path: str, output_dir: str = "_data/news",
                     method: str = "finbert", delay: float = 0.3):
    """전체 뉴스 감성 분석"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 로드
    print(f"📂 뉴스 데이터 로드: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    print(f"   총 {len(df):,}건")
    
    # 분석기 초기화
    analyzer = NewsAnalyzer(method=method)
    
    # 분석 실행
    df_analyzed = analyzer.analyze_dataframe(df, delay=delay)
    
    # 분석 결과 저장
    analyzed_path = input_path.replace('.csv', f'_analyzed_{method}.csv')
    df_analyzed.to_csv(analyzed_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 분석 결과 저장: {analyzed_path}")
    
    # 일별 집계
    df_daily = analyzer.aggregate_daily(df_analyzed)
    
    daily_path = f"{output_dir}/news_sentiment_daily.csv"
    df_daily.to_csv(daily_path, index=False, encoding='utf-8-sig')
    print(f"💾 일별 집계 저장: {daily_path}")
    
    return df_daily


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="뉴스 감성 분석")
    parser.add_argument("-i", "--input", type=str, required=True, help="입력 CSV 파일")
    parser.add_argument("-o", "--output", type=str, default="_data/news", help="출력 디렉토리")
    parser.add_argument("--method", type=str, default="finbert", 
                        choices=["llm", "finbert"], help="분석 방식")
    parser.add_argument("--delay", type=float, default=0.3, help="API 호출 간격 (llm)")
    
    args = parser.parse_args()
    
    analyze_all_news(
        input_path=args.input,
        output_dir=args.output,
        method=args.method,
        delay=args.delay
    )

