"""
LLM(GPT/Claude/Ollama)을 사용한 뉴스 감성 분석 모듈
- 감성 점수 (-1 ~ +1)
- 영향 강도 (1~5)
- 이벤트 유형 추출
"""
import pandas as pd
import numpy as np
import json
import os
import time
from typing import List, Dict, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# .env 파일 로드
try:
    from dotenv import load_dotenv
    # 프로젝트 루트의 .env 파일 로드
    env_paths = [
        'D:/stock/.env',  # 재현님 .env 위치
        os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'),
        '.env',
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"✅ .env 파일 로드: {env_path}")
            break
    else:
        load_dotenv()  # 기본 위치
except ImportError:
    print("⚠️ python-dotenv 패키지가 없습니다. pip install python-dotenv")


class LLMNewsRefiner:
    """LLM을 사용한 뉴스 감성 분석 클래스"""
    
    def __init__(self, provider: str = "openai", model: str = None, api_key: str = None):
        """
        LLM 초기화
        
        Args:
            provider: "openai", "anthropic", "ollama" 중 선택
            model: 모델명 (기본값: provider별 기본 모델)
            api_key: API 키 (환경변수에서 자동 로드 가능)
        """
        self.provider = provider
        self.client = None
        
        if provider == "openai":
            self.model = model or "gpt-4o-mini"  # 비용 효율적
            self._init_openai(api_key)
        elif provider == "anthropic":
            self.model = model or "claude-3-haiku-20240307"  # 비용 효율적
            self._init_anthropic(api_key)
        elif provider == "ollama":
            self.model = model or "llama3.1"  # 로컬 무료
            self._init_ollama()
        else:
            raise ValueError(f"지원하지 않는 provider: {provider}")
    
    def _init_openai(self, api_key: str = None):
        """OpenAI 클라이언트 초기화"""
        try:
            from openai import OpenAI
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
            self.client = OpenAI(api_key=api_key)
            print(f"✅ OpenAI 초기화 완료 (모델: {self.model})")
        except ImportError:
            print("❌ openai 패키지가 설치되지 않았습니다.")
            print("   pip install openai")
            raise
    
    def _init_anthropic(self, api_key: str = None):
        """Anthropic 클라이언트 초기화"""
        try:
            import anthropic
            api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY가 설정되지 않았습니다.")
            self.client = anthropic.Anthropic(api_key=api_key)
            print(f"✅ Anthropic 초기화 완료 (모델: {self.model})")
        except ImportError:
            print("❌ anthropic 패키지가 설치되지 않았습니다.")
            print("   pip install anthropic")
            raise
    
    def _init_ollama(self):
        """Ollama 클라이언트 초기화 (로컬)"""
        try:
            import requests
            # Ollama 서버 확인
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Ollama 초기화 완료 (모델: {self.model})")
                self.client = "ollama"
            else:
                raise ConnectionError("Ollama 서버에 연결할 수 없습니다.")
        except Exception as e:
            print(f"❌ Ollama 연결 실패: {e}")
            print("   ollama serve 명령으로 서버를 시작하세요.")
            raise
    
    def _create_prompt(self, news_title: str, stock_name: str = None) -> str:
        """분석 프롬프트 생성"""
        stock_context = f"'{stock_name}' 종목에 대한 " if stock_name else ""
        
        prompt = f"""다음은 {stock_context}뉴스 제목입니다. 주가에 미칠 영향을 분석해주세요.

뉴스 제목: "{news_title}"

다음 JSON 형식으로만 응답하세요:
{{
    "sentiment": 0.0,      // 감성 점수 (-1.0 ~ +1.0, 소수점 2자리)
    "impact": 3,           // 영향 강도 (1~5, 정수)
    "event_type": "일반",  // 이벤트 유형
    "reason": "분석 이유"  // 한 줄 설명
}}

이벤트 유형 목록:
- 실적: 실적 발표, 매출, 영업이익 관련
- 배당: 배당금, 배당 정책 관련
- 계약: 대규모 계약, 수주, 공급 계약
- 투자: 시설 투자, R&D, 인수합병
- 인사: 경영진 변동, 조직 개편
- 규제: 정부 정책, 규제, 법적 이슈
- 시장: 업황, 경쟁, 시장 트렌드
- 일반: 기타

감성 점수 기준:
- +0.8 ~ +1.0: 매우 긍정 (사상 최대 실적, 대규모 계약 등)
- +0.4 ~ +0.7: 긍정 (실적 개선, 신사업 진출 등)
- -0.3 ~ +0.3: 중립 (일반 뉴스, 영향 불확실)
- -0.7 ~ -0.4: 부정 (실적 악화, 소송 등)
- -1.0 ~ -0.8: 매우 부정 (대규모 손실, 중대 사고 등)

JSON만 출력하세요:"""
        
        return prompt
    
    def _parse_response(self, response_text: str) -> Dict:
        """LLM 응답 파싱"""
        try:
            # JSON 추출 시도
            # 응답에서 JSON 부분만 추출
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                
                # 값 검증 및 클리핑
                result['sentiment'] = max(-1.0, min(1.0, float(result.get('sentiment', 0))))
                result['impact'] = max(1, min(5, int(result.get('impact', 3))))
                result['event_type'] = result.get('event_type', '일반')
                result['reason'] = result.get('reason', '')
                
                return result
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"⚠️ 응답 파싱 오류: {e}")
        
        # 기본값 반환
        return {
            'sentiment': 0.0,
            'impact': 3,
            'event_type': '일반',
            'reason': '파싱 실패'
        }
    
    def analyze_single(self, news_title: str, stock_name: str = None) -> Dict:
        """단일 뉴스 분석"""
        prompt = self._create_prompt(news_title, stock_name)
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200
                )
                response_text = response.choices[0].message.content
                
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text
                
            elif self.provider == "ollama":
                import requests
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1}
                    },
                    timeout=30
                )
                response_text = response.json()['response']
            
            return self._parse_response(response_text)
            
        except Exception as e:
            print(f"⚠️ API 호출 오류: {e}")
            return {
                'sentiment': 0.0,
                'impact': 3,
                'event_type': '일반',
                'reason': f'API 오류: {str(e)}'
            }
    
    def analyze_batch(self, news_list: List[Dict], stock_name: str = None, 
                      delay: float = 0.5, show_progress: bool = True) -> List[Dict]:
        """
        배치 뉴스 분석
        
        Args:
            news_list: [{"title": "뉴스제목", "date": "2026-01-01"}, ...]
            stock_name: 종목명
            delay: API 호출 간격 (초)
            show_progress: 진행률 표시 여부
            
        Returns:
            분석 결과 리스트
        """
        results = []
        total = len(news_list)
        
        for i, news in enumerate(news_list):
            title = news.get('title', '')
            date = news.get('date', '')
            
            if not title:
                results.append({
                    'date': date,
                    'title': title,
                    'sentiment': 0.0,
                    'impact': 1,
                    'event_type': '일반',
                    'reason': '제목 없음'
                })
                continue
            
            # 분석
            analysis = self.analyze_single(title, stock_name)
            analysis['date'] = date
            analysis['title'] = title
            results.append(analysis)
            
            # 진행률 표시
            if show_progress and (i + 1) % 10 == 0:
                print(f"  진행률: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")
            
            # API 속도 제한 방지
            if delay > 0 and i < total - 1:
                time.sleep(delay)
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, 
                          title_column: str = "제목",
                          date_column: str = "날짜",
                          stock_name: str = None,
                          delay: float = 0.5) -> pd.DataFrame:
        """
        DataFrame 분석
        
        Args:
            df: 뉴스 DataFrame
            title_column: 제목 컬럼명
            date_column: 날짜 컬럼명
            stock_name: 종목명
            delay: API 호출 간격
            
        Returns:
            분석 결과가 추가된 DataFrame
        """
        print(f"\n📊 LLM 뉴스 분석 시작")
        print(f"   총 {len(df)}건의 뉴스")
        print(f"   모델: {self.provider}/{self.model}")
        print(f"   예상 시간: {len(df) * delay / 60:.1f}분\n")
        
        # 뉴스 리스트 생성
        news_list = []
        for _, row in df.iterrows():
            news_list.append({
                'title': str(row.get(title_column, '')),
                'date': str(row.get(date_column, ''))
            })
        
        # 배치 분석
        results = self.analyze_batch(news_list, stock_name, delay)
        
        # 결과를 DataFrame에 추가
        df_result = df.copy()
        df_result['llm_sentiment'] = [r['sentiment'] for r in results]
        df_result['llm_impact'] = [r['impact'] for r in results]
        df_result['llm_event_type'] = [r['event_type'] for r in results]
        df_result['llm_reason'] = [r['reason'] for r in results]
        
        # 결과 통계
        print(f"\n✅ 분석 완료!")
        print(f"   평균 감성: {df_result['llm_sentiment'].mean():.3f}")
        print(f"   긍정 뉴스: {len(df_result[df_result['llm_sentiment'] > 0.3])}건")
        print(f"   부정 뉴스: {len(df_result[df_result['llm_sentiment'] < -0.3])}건")
        
        # 이벤트 유형 분포
        print(f"\n   이벤트 유형 분포:")
        for event_type, count in df_result['llm_event_type'].value_counts().items():
            print(f"   - {event_type}: {count}건")
        
        return df_result
    
    def aggregate_daily(self, df: pd.DataFrame, 
                        date_column: str = "날짜",
                        sentiment_column: str = "llm_sentiment",
                        impact_column: str = "llm_impact") -> pd.DataFrame:
        """
        날짜별 감성 점수 집계
        
        Args:
            df: 분석된 DataFrame
            date_column: 날짜 컬럼명
            sentiment_column: 감성 점수 컬럼명
            impact_column: 영향 강도 컬럼명
            
        Returns:
            날짜별 집계 DataFrame
        """
        # 영향 강도로 가중 평균
        df_temp = df.copy()
        df_temp['weighted_sentiment'] = df_temp[sentiment_column] * df_temp[impact_column]
        
        # 날짜별 집계
        df_daily = df_temp.groupby(date_column).agg({
            'weighted_sentiment': 'sum',
            impact_column: 'sum',
            sentiment_column: ['mean', 'count']
        }).reset_index()
        
        # 컬럼명 정리
        df_daily.columns = [date_column, 'weighted_sum', 'impact_sum', 'sentiment_mean', 'news_count']
        
        # 가중 평균 계산
        df_daily['sentiment_weighted'] = df_daily['weighted_sum'] / df_daily['impact_sum']
        
        # 최종 정리
        df_daily = df_daily[[date_column, 'sentiment_weighted', 'sentiment_mean', 'news_count']]
        df_daily = df_daily.rename(columns={
            'sentiment_weighted': 'sentiment_score',
            'sentiment_mean': 'sentiment_simple'
        })
        
        # 소수점 정리
        df_daily['sentiment_score'] = df_daily['sentiment_score'].round(4)
        df_daily['sentiment_simple'] = df_daily['sentiment_simple'].round(4)
        
        # 날짜 정렬
        df_daily = df_daily.sort_values(date_column).reset_index(drop=True)
        
        print(f"\n📊 날짜별 집계 완료")
        print(f"   총 {len(df_daily)}일")
        print(f"   평균 감성 (가중): {df_daily['sentiment_score'].mean():.4f}")
        print(f"   평균 감성 (단순): {df_daily['sentiment_simple'].mean():.4f}")
        
        return df_daily


def process_news_with_llm(
    input_path: str,
    output_path: str = None,
    provider: str = "openai",
    model: str = None,
    api_key: str = None,
    title_column: str = "HTS_공시_제목_내용",
    date_column: str = "작성일자",
    stock_name: str = None,
    delay: float = 0.5,
    aggregate: bool = True
) -> pd.DataFrame:
    """
    뉴스 파일을 LLM으로 분석하는 함수
    
    Args:
        input_path: 입력 CSV 파일 경로
        output_path: 출력 CSV 파일 경로
        provider: LLM 제공자 (openai, anthropic, ollama)
        model: 모델명
        api_key: API 키
        title_column: 제목 컬럼명
        date_column: 날짜 컬럼명
        stock_name: 종목명
        delay: API 호출 간격 (초)
        aggregate: 날짜별 집계 여부
        
    Returns:
        분석 결과 DataFrame
    """
    # 파일 읽기
    print(f"📂 파일 읽기: {input_path}")
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(input_path, encoding='cp949')
    print(f"   총 {len(df)}건 로드")
    
    # LLM 초기화
    refiner = LLMNewsRefiner(provider=provider, model=model, api_key=api_key)
    
    # 분석
    df_result = refiner.analyze_dataframe(
        df,
        title_column=title_column,
        date_column=date_column,
        stock_name=stock_name,
        delay=delay
    )
    
    # 결과 저장
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_llm.csv"
    
    df_result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 결과 저장: {output_path}")
    
    # 날짜별 집계
    if aggregate:
        # 날짜 형식 변환
        if date_column in df_result.columns:
            dates = df_result[date_column].astype(str)
            df_result['날짜'] = dates.apply(
                lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if len(x) == 8 and x.isdigit() else x
            )
        
        df_daily = refiner.aggregate_daily(df_result, date_column='날짜')
        
        daily_path = output_path.replace('.csv', '_daily.csv')
        df_daily.to_csv(daily_path, index=False, encoding='utf-8-sig')
        print(f"💾 일별 집계 저장: {daily_path}")
        
        return df_daily
    
    return df_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM 기반 뉴스 감성 분석")
    parser.add_argument("-i", "--input", type=str, required=True, help="입력 CSV 파일")
    parser.add_argument("-o", "--output", type=str, default=None, help="출력 CSV 파일")
    parser.add_argument("--provider", type=str, default="openai", 
                        choices=["openai", "anthropic", "ollama"], help="LLM 제공자")
    parser.add_argument("--model", type=str, default=None, help="모델명")
    parser.add_argument("--api-key", type=str, default=None, help="API 키")
    parser.add_argument("--title-column", type=str, default="HTS_공시_제목_내용", help="제목 컬럼명")
    parser.add_argument("--date-column", type=str, default="작성일자", help="날짜 컬럼명")
    parser.add_argument("--stock-name", type=str, default=None, help="종목명")
    parser.add_argument("--delay", type=float, default=0.5, help="API 호출 간격 (초)")
    parser.add_argument("--no-aggregate", action="store_true", help="날짜별 집계 안 함")
    
    args = parser.parse_args()
    
    df_result = process_news_with_llm(
        input_path=args.input,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        title_column=args.title_column,
        date_column=args.date_column,
        stock_name=args.stock_name,
        delay=args.delay,
        aggregate=not args.no_aggregate
    )
    
    print("\n" + "=" * 60)
    print("결과 미리보기")
    print("=" * 60)
    print(df_result.head(10))

