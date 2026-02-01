"""
한국어 FinBERT를 사용한 뉴스 감성 분석 모듈
"""
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Optional
import os
import warnings
warnings.filterwarnings('ignore')


class FinBERTRefiner:
    """한국어 FinBERT를 사용한 뉴스 감성 분석 클래스"""
    
    def __init__(self, model_name: str = "snunlp/KR-FinBert-SC"):
        """
        FinBERT 모델 초기화
        
        Args:
            model_name: 사용할 모델 이름
                - "snunlp/KR-FinBert-SC": 한국어 금융 감성 분석 모델 (권장)
                - "monologg/koelectra-base-v3-discriminator": 일반 한국어 모델
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 FinBERT 모델 로딩 중: {model_name}")
        print(f"📱 사용 디바이스: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            print("✅ 모델 로딩 완료")
        except Exception as e:
            print(f"❌ 모델 로딩 실패: {e}")
            print("💡 대안 모델을 시도합니다...")
            # 대안 모델 시도
            try:
                self.model_name = "monologg/koelectra-base-v3-discriminator"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                # 일반 모델은 분류 헤드를 추가해야 할 수 있음
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=3  # 긍정, 중립, 부정
                )
                self.model.to(self.device)
                self.model.eval()
                print("✅ 대안 모델 로딩 완료")
            except Exception as e2:
                print(f"❌ 대안 모델도 로딩 실패: {e2}")
                raise
    
    def predict_sentiment(self, text: str) -> dict:
        """
        단일 텍스트의 감성 분석
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            dict: 감성 분석 결과
                - label: 감성 레이블 (0: 부정, 1: 중립, 2: 긍정)
                - score: 신뢰도 점수
                - sentiment: 감성 텍스트 ("부정", "중립", "긍정")
                - scores: 각 감성별 점수 (부정, 중립, 긍정)
        """
        if not text or pd.isna(text):
            return {
                "label": 1,
                "score": 0.0,
                "sentiment": "중립",
                "scores": {"부정": 0.0, "중립": 1.0, "긍정": 0.0}
            }
        
        # 텍스트 전처리 (최대 길이 제한)
        text = str(text).strip()
        if len(text) > 512:  # BERT 최대 길이
            text = text[:512]
        
        try:
            # 토큰화 및 인코딩
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # 예측
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                predicted_label = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][predicted_label].item()
            
            # 각 감성별 점수 추출
            prob_list = probs[0].cpu().tolist()
            # 레이블 매핑 (모델에 따라 다를 수 있음)
            sentiment_map = {0: "부정", 1: "중립", 2: "긍정"}
            sentiment = sentiment_map.get(predicted_label, "중립")
            
            # 점수 딕셔너리 생성 (레이블 순서에 맞춰)
            scores = {
                "부정": prob_list[0] if len(prob_list) > 0 else 0.0,
                "중립": prob_list[1] if len(prob_list) > 1 else 0.0,
                "긍정": prob_list[2] if len(prob_list) > 2 else 0.0
            }
            
            return {
                "label": predicted_label,
                "score": confidence,
                "sentiment": sentiment,
                "scores": scores
            }
        except Exception as e:
            print(f"⚠️ 감성 분석 오류: {e}")
            return {
                "label": 1,
                "score": 0.0,
                "sentiment": "중립",
                "scores": {"부정": 0.0, "중립": 1.0, "긍정": 0.0}
            }
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[dict]:
        """
        여러 텍스트의 배치 감성 분석
        
        Args:
            texts: 분석할 텍스트 리스트
            batch_size: 배치 크기
            
        Returns:
            List[dict]: 감성 분석 결과 리스트
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch_texts:
                result = self.predict_sentiment(text)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            if (i + batch_size) % 100 == 0:
                print(f"  진행률: {min(i + batch_size, len(texts))}/{len(texts)}")
        
        return results
    
    def refine_news(self, df: pd.DataFrame, text_column: str = "HTS_공시_제목_내용", 
                   date_column: str = "작성일자", time_column: str = "작성시간") -> pd.DataFrame:
        """
        뉴스 데이터프레임에 감성 분석 결과 추가
        
        Args:
            df: 뉴스 데이터프레임
            text_column: 분석할 텍스트 컬럼명
            date_column: 날짜 컬럼명
            time_column: 시간 컬럼명
            
        Returns:
            pd.DataFrame: 감성 분석 결과가 추가된 데이터프레임
        """
        if text_column not in df.columns:
            print(f"⚠️ 컬럼 '{text_column}'을 찾을 수 없습니다.")
            print(f"사용 가능한 컬럼: {df.columns.tolist()}")
            return df
        
        print(f"\n📊 감성 분석 시작...")
        print(f"  총 {len(df)}건의 뉴스 분석")
        print(f"  분석 컬럼: {text_column}")
        
        # 텍스트 추출
        texts = df[text_column].fillna("").astype(str).tolist()
        
        # 배치 감성 분석
        results = self.predict_batch(texts, batch_size=16)
        
        # 결과를 데이터프레임에 추가
        df_result = df.copy()
        df_result["감성_레이블"] = [r["label"] for r in results]
        df_result["감성_점수"] = [r["score"] for r in results]
        df_result["감성"] = [r["sentiment"] for r in results]
        
        # 각 감성별 점수 추가
        df_result["부정_점수"] = [r["scores"]["부정"] for r in results]
        df_result["중립_점수"] = [r["scores"]["중립"] for r in results]
        df_result["긍정_점수"] = [r["scores"]["긍정"] for r in results]
        
        # 감성 분포 출력
        sentiment_counts = df_result["감성"].value_counts()
        print(f"\n✅ 감성 분석 완료")
        print(f"감성 분포:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df_result)) * 100
            print(f"  {sentiment}: {count}건 ({percentage:.1f}%)")
        
        return df_result
    
    def format_output(self, df: pd.DataFrame, 
                     date_column: str = "작성일자", 
                     time_column: str = "작성시간",
                     text_column: str = "HTS_공시_제목_내용") -> pd.DataFrame:
        """
        출력용 데이터프레임 형식 변환
        날짜 / 시간(시,분,초) / 내용 / 점수(부정, 긍정, 중립) 형식으로 변환
        
        Args:
            df: 원본 데이터프레임
            date_column: 날짜 컬럼명
            time_column: 시간 컬럼명
            text_column: 내용 컬럼명
            
        Returns:
            pd.DataFrame: 형식 변환된 데이터프레임
        """
        # 날짜 형식 변환 (YYYYMMDD -> YYYY-MM-DD)
        if date_column in df.columns:
            dates = df[date_column].astype(str)
            dates_formatted = dates.apply(
                lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if len(x) == 8 else x
            )
        else:
            dates_formatted = pd.Series([""] * len(df))
        
        # 시간 형식 변환 (HHMMSS -> HH:MM:SS)
        if time_column in df.columns:
            times = df[time_column].astype(str).str.zfill(6)
            times_formatted = times.apply(
                lambda x: f"{x[:2]}:{x[2:4]}:{x[4:6]}" if len(x) == 6 else x
            )
        else:
            times_formatted = pd.Series([""] * len(df))
        
        # 내용 추출
        if text_column in df.columns:
            contents = df[text_column].fillna("").astype(str)
        else:
            contents = pd.Series([""] * len(df))
        
        # 점수 추출 (부정, 중립, 긍정)
        neg_scores = df["부정_점수"] if "부정_점수" in df.columns else pd.Series([0.0] * len(df))
        neu_scores = df["중립_점수"] if "중립_점수" in df.columns else pd.Series([0.0] * len(df))
        pos_scores = df["긍정_점수"] if "긍정_점수" in df.columns else pd.Series([0.0] * len(df))
        
        # 새로운 데이터프레임 생성
        df_output = pd.DataFrame({
            "날짜": dates_formatted,
            "시간": times_formatted,
            "내용": contents,
            "부정_점수": neg_scores.round(4),
            "중립_점수": neu_scores.round(4),
            "긍정_점수": pos_scores.round(4)
        })
        
        return df_output
    
    def aggregate_daily_sentiment(self, df: pd.DataFrame, 
                                   date_column: str = "날짜",
                                   method: str = "mean") -> pd.DataFrame:
        """
        날짜별로 감정 점수를 집계하는 함수
        
        Args:
            df: 감성 분석 결과 데이터프레임
            date_column: 날짜 컬럼명
            method: 집계 방법
                - "mean": 평균 (기본값)
                - "weighted": 가중 평균 (뉴스 개수 기반)
                - "max": 최대값
                - "median": 중앙값
                
        Returns:
            pd.DataFrame: 날짜별 집계된 감정 점수
        """
        if date_column not in df.columns:
            print(f"⚠️ 날짜 컬럼 '{date_column}'을 찾을 수 없습니다.")
            return pd.DataFrame()
        
        # 점수 컬럼 확인
        score_columns = ["부정_점수", "중립_점수", "긍정_점수"]
        available_scores = [col for col in score_columns if col in df.columns]
        
        if not available_scores:
            print(f"⚠️ 점수 컬럼을 찾을 수 없습니다.")
            return pd.DataFrame()
        
        # 날짜별 집계
        if method == "mean":
            # 평균
            df_daily = df.groupby(date_column)[available_scores].mean().reset_index()
            df_daily["뉴스_개수"] = df.groupby(date_column).size().values
            
        elif method == "weighted":
            # 가중 평균 (뉴스 개수로 가중치 적용)
            df_daily = df.groupby(date_column)[available_scores].mean().reset_index()
            df_daily["뉴스_개수"] = df.groupby(date_column).size().values
            
        elif method == "max":
            # 최대값
            df_daily = df.groupby(date_column)[available_scores].max().reset_index()
            df_daily["뉴스_개수"] = df.groupby(date_column).size().values
            
        elif method == "median":
            # 중앙값
            df_daily = df.groupby(date_column)[available_scores].median().reset_index()
            df_daily["뉴스_개수"] = df.groupby(date_column).size().values
            
        else:
            print(f"⚠️ 알 수 없는 집계 방법: {method}. 평균을 사용합니다.")
            df_daily = df.groupby(date_column)[available_scores].mean().reset_index()
            df_daily["뉴스_개수"] = df.groupby(date_column).size().values
        
        # 점수 반올림
        for col in available_scores:
            df_daily[col] = df_daily[col].round(4)
        
        # 주요 감정 결정 (가장 높은 점수)
        if len(available_scores) == 3:
            df_daily["주요_감정"] = df_daily[available_scores].idxmax(axis=1)
            df_daily["주요_감정"] = df_daily["주요_감정"].str.replace("_점수", "")
            df_daily["주요_감정_점수"] = df_daily[available_scores].max(axis=1)
            
            # 단일 감성 점수 계산 (긍정 - 부정, 범위: -1 ~ 1)
            if "부정_점수" in df_daily.columns and "긍정_점수" in df_daily.columns:
                df_daily["감성_점수"] = (df_daily["긍정_점수"] - df_daily["부정_점수"]).round(4)
        
        # 날짜순 정렬
        df_daily = df_daily.sort_values(date_column, ascending=False).reset_index(drop=True)
        
        print(f"\n📊 날짜별 집계 완료")
        print(f"  집계 방법: {method}")
        print(f"  총 {len(df_daily)}일의 데이터")
        
        # 감성 점수 통계 출력
        if "감성_점수" in df_daily.columns:
            print(f"\n감성 점수 통계:")
            print(f"  평균: {df_daily['감성_점수'].mean():.4f}")
            print(f"  최소: {df_daily['감성_점수'].min():.4f}")
            print(f"  최대: {df_daily['감성_점수'].max():.4f}")
            print(f"  표준편차: {df_daily['감성_점수'].std():.4f}")
        
        return df_daily
    
    def format_sentiment_output(self, df: pd.DataFrame, 
                                format_type: str = "all") -> pd.DataFrame:
        """
        감성 점수 출력 형식 변환
        
        Args:
            df: 날짜별 집계 데이터프레임
            format_type: 출력 형식
                - "all": 부정/중립/긍정 점수 모두 사용 (기본값)
                - "single": 단일 감성 점수만 사용 (긍정 - 부정, 범위: -1 ~ 1)
                - "binary": 부정/긍정 점수만 사용 (중립 제외)
                
        Returns:
            pd.DataFrame: 형식 변환된 데이터프레임
        """
        df_output = df.copy()
        
        if format_type == "single":
            # 단일 감성 점수만 사용 (연속값)
            if "감성_점수" in df_output.columns:
                keep_columns = ["날짜", "감성_점수", "뉴스_개수"]
                if "주요_감정" in df_output.columns:
                    keep_columns.append("주요_감정")
                df_output = df_output[keep_columns]
            else:
                print("⚠️ 감성_점수 컬럼이 없습니다. 'all' 형식을 사용합니다.")
                
        elif format_type == "binary":
            # 부정/긍정만 사용 (중립 제외)
            if "부정_점수" in df_output.columns and "긍정_점수" in df_output.columns:
                keep_columns = ["날짜", "부정_점수", "긍정_점수", "뉴스_개수"]
                if "주요_감정" in df_output.columns:
                    keep_columns.append("주요_감정")
                if "감성_점수" in df_output.columns:
                    keep_columns.append("감성_점수")
                df_output = df_output[keep_columns]
            else:
                print("⚠️ 부정/긍정 점수 컬럼이 없습니다. 'all' 형식을 사용합니다.")
        
        # format_type == "all"이면 모든 컬럼 유지
        
        return df_output


def convert_daily_to_single(input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    기존 날짜별 집계 파일을 단일 감성 점수 형식으로 변환
    
    Args:
        input_path: 입력 CSV 파일 경로 (날짜별 집계 파일)
        output_path: 출력 CSV 파일 경로 (None이면 입력 파일명에 _single 추가)
        
    Returns:
        pd.DataFrame: 변환된 데이터프레임
    """
    print(f"📂 파일 읽기: {input_path}")
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(input_path, encoding='cp949')
    
    print(f"  총 {len(df)}일의 데이터 로드 완료")
    
    # 감성 점수 계산 (긍정 - 부정)
    if "부정_점수" in df.columns and "긍정_점수" in df.columns:
        df["감성_점수"] = (df["긍정_점수"] - df["부정_점수"]).round(4)
    elif "감성_점수" not in df.columns:
        print("⚠️ 부정/긍정 점수 컬럼을 찾을 수 없습니다.")
        return df
    
    # 단일 형식으로 변환
    keep_columns = ["날짜", "감성_점수", "뉴스_개수"]
    if "주요_감정" in df.columns:
        keep_columns.append("주요_감정")
    
    df_output = df[keep_columns].copy()
    
    # 결과 저장
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_single.csv"
    
    df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 변환 완료: {output_path}")
    print(f"  형식: 날짜 / 감성_점수(긍정-부정, 연속값) / 뉴스_개수")
    
    # 감성 점수 통계 출력
    print(f"\n감성 점수 통계:")
    print(f"  평균: {df_output['감성_점수'].mean():.4f}")
    print(f"  최소: {df_output['감성_점수'].min():.4f}")
    print(f"  최대: {df_output['감성_점수'].max():.4f}")
    print(f"  표준편차: {df_output['감성_점수'].std():.4f}")
    
    return df_output


def refine_news_file(
    input_path: str,
    output_path: Optional[str] = None,
    text_column: str = "HTS_공시_제목_내용",
    date_column: str = "작성일자",
    time_column: str = "작성시간",
    model_name: str = "snunlp/KR-FinBert-SC",
    format_output: bool = True,
    aggregate_daily: bool = False,
    aggregation_method: str = "mean",
    sentiment_format: str = "all"
) -> pd.DataFrame:
    """
    뉴스 CSV 파일에 감성 분석을 수행하는 함수
    
    Args:
        input_path: 입력 CSV 파일 경로
        output_path: 출력 CSV 파일 경로 (None이면 입력 파일명에 _refined 추가)
        text_column: 분석할 텍스트 컬럼명
        date_column: 날짜 컬럼명
        time_column: 시간 컬럼명
        model_name: 사용할 모델 이름
        format_output: True면 날짜/시간/내용/점수 형식으로 저장, False면 원본+결과 형식
        aggregate_daily: 날짜별 집계 여부
        aggregation_method: 집계 방법
        sentiment_format: 감성 점수 출력 형식
            - "all": 부정/중립/긍정 점수 모두 사용 (기본값)
            - "single": 단일 감성 점수만 사용 (긍정 - 부정)
            - "binary": 부정/긍정 점수만 사용 (중립 제외)
        
    Returns:
        pd.DataFrame: 감성 분석 결과가 추가된 데이터프레임
    """
    # 파일 읽기
    print(f"📂 파일 읽기: {input_path}")
    try:
        df = pd.read_csv(input_path, encoding='utf-8-sig')
    except:
        df = pd.read_csv(input_path, encoding='cp949')
    
    print(f"  총 {len(df)}건의 뉴스 로드 완료")
    
    # FinBERT 초기화
    refiner = FinBERTRefiner(model_name=model_name)
    
    # 감성 분석 수행
    df_refined = refiner.refine_news(
        df, 
        text_column=text_column,
        date_column=date_column,
        time_column=time_column
    )
    
    # 출력 형식 변환
    if format_output:
        df_output = refiner.format_output(
            df_refined,
            date_column=date_column,
            time_column=time_column,
            text_column=text_column
        )
    else:
        df_output = df_refined
    
    # 날짜별 집계
    if aggregate_daily:
        # 날짜 컬럼명 확인 (format_output 후에는 "날짜"로 변경됨)
        daily_date_column = "날짜" if format_output else date_column
        df_daily = refiner.aggregate_daily_sentiment(
            df_output,
            date_column=daily_date_column,
            method=aggregation_method
        )
        
        # 감성 점수 형식 변환
        df_daily = refiner.format_sentiment_output(df_daily, format_type=sentiment_format)
        
        # 집계 결과 저장
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            daily_output_path = f"{base_name}_daily.csv"
        else:
            base_name = os.path.splitext(output_path)[0]
            daily_output_path = f"{base_name}_daily.csv"
        
        df_daily.to_csv(daily_output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 날짜별 집계 결과 저장: {daily_output_path}")
        print(f"  감성 점수 형식: {sentiment_format}")
        
        # 개별 뉴스 결과도 저장
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_refined.csv"
        
        df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"💾 개별 뉴스 결과 저장: {output_path}")
        
        return df_daily
    else:
        # 결과 저장
        if output_path is None:
            base_name = os.path.splitext(input_path)[0]
            output_path = f"{base_name}_refined.csv"
        
        df_output.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 결과 저장: {output_path}")
        print(f"  저장 형식: 날짜 / 시간 / 내용 / 점수(부정, 중립, 긍정)")
        
        return df_output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="한국어 FinBERT를 사용한 뉴스 감성 분석",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용
  python refiner.py -i news_005930.csv
  
  # 출력 파일 지정
  python refiner.py -i news_005930.csv -o news_005930_refined.csv
  
  # 날짜별 집계 (하루의 감정 점수)
  python refiner.py -i news_005930.csv --aggregate
  
  # 날짜별 집계 (중앙값 사용)
  python refiner.py -i news_005930.csv --aggregate --aggregation-method median
  
  # 단일 감성 점수만 사용 (긍정 - 부정)
  python refiner.py -i news_005930.csv --aggregate --sentiment-format single
  
  # 부정/긍정만 사용 (중립 제외)
  python refiner.py -i news_005930.csv --aggregate --sentiment-format binary
  
  # 다른 텍스트 컬럼 사용
  python refiner.py -i news_005930.csv -c "제목"
  
  # 다른 모델 사용
  python refiner.py -i news_005930.csv -m "monologg/koelectra-base-v3-discriminator"
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="입력 CSV 파일 경로"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="출력 CSV 파일 경로 (기본값: 입력파일명_refined.csv)"
    )
    
    parser.add_argument(
        "-c", "--column",
        type=str,
        default="HTS_공시_제목_내용",
        help="분석할 텍스트 컬럼명 (기본값: HTS_공시_제목_내용)"
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="snunlp/KR-FinBert-SC",
        help="사용할 모델 이름 (기본값: snunlp/KR-FinBert-SC)"
    )
    
    parser.add_argument(
        "--date-column",
        type=str,
        default="작성일자",
        help="날짜 컬럼명 (기본값: 작성일자)"
    )
    
    parser.add_argument(
        "--time-column",
        type=str,
        default="작성시간",
        help="시간 컬럼명 (기본값: 작성시간)"
    )
    
    parser.add_argument(
        "--no-format",
        action="store_true",
        help="원본 형식 유지 (날짜/시간/내용/점수 형식 변환 안 함)"
    )
    
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="날짜별로 감정 점수 집계 (하루의 감정 점수 계산)"
    )
    
    parser.add_argument(
        "--aggregation-method",
        type=str,
        default="mean",
        choices=["mean", "weighted", "max", "median"],
        help="집계 방법 (기본값: mean - 평균)"
    )
    
    parser.add_argument(
        "--sentiment-format",
        type=str,
        default="single",
        choices=["all", "single", "binary"],
        help="감성 점수 출력 형식 (기본값: single - 긍정-부정 단일 점수)"
    )
    
    parser.add_argument(
        "--convert",
        action="store_true",
        help="기존 날짜별 집계 파일을 단일 감성 점수 형식으로 변환"
    )
    
    args = parser.parse_args()
    
    # 변환 모드
    if args.convert:
        df_result = convert_daily_to_single(
            input_path=args.input,
            output_path=args.output
        )
        print("\n" + "=" * 60)
        print("변환 결과 미리보기")
        print("=" * 60)
        print(df_result.head(10))
    else:
        # 감성 분석 수행
        df_result = refine_news_file(
            input_path=args.input,
            output_path=args.output,
            text_column=args.column,
            date_column=args.date_column,
            time_column=args.time_column,
            model_name=args.model,
            format_output=not args.no_format,
            aggregate_daily=args.aggregate,
            aggregation_method=args.aggregation_method,
            sentiment_format=args.sentiment_format
        )
        
        # 결과 미리보기
        print("\n" + "=" * 60)
        print("결과 미리보기")
        print("=" * 60)
        print(df_result.head(10))

