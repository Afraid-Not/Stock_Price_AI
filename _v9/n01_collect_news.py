"""
30개 종목 뉴스 수집 스크립트
한투 API를 사용하여 최근 1년치 뉴스를 수집합니다.
"""
import pandas as pd
import requests
import json
import time
import os
import sys
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

# .env 로드
load_dotenv('D:/stock/.env')

# s00_get_token import
from s00_get_token import get_access_token

# 설정
APP_KEY = os.getenv("REAL_APP_KEY")
APP_SECRET = os.getenv("REAL_APP_SECRET")
BASE_URL = "https://openapi.koreainvestment.com:9443"
TR_ID_NEWS_TITLE = "FHKST01011800"

# 컬럼 매핑
COLUMN_MAPPING = {
    'cntt_usiq_srno': '내용_조회용_일련번호',
    'news_ofer_entp_code': '뉴스_제공_업체_코드',
    'data_dt': '작성일자',
    'data_tm': '작성시간',
    'hts_pbnt_titl_cntt': 'HTS_공시_제목_내용',
    'news_lrdv_code': '뉴스_대구분',
    'dorg': '자료원',
    'iscd1': '종목코드1',
}


def get_news_by_date(token: str, stock_code: str, date: str, max_depth: int = 20, max_retries: int = 3) -> pd.DataFrame:
    """특정 날짜의 종목 뉴스 수집 (재시도 로직 포함)"""
    path = "/uapi/domestic-stock/v1/quotations/news-title"
    url = f"{BASE_URL}{path}"
    
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": TR_ID_NEWS_TITLE
    }
    
    all_data = []
    input_srno = ""
    
    for depth in range(max_depth):
        params = {
            "FID_NEWS_OFER_ENTP_CODE": "",
            "FID_COND_MRKT_CLS_CODE": "",
            "FID_INPUT_ISCD": stock_code,
            "FID_TITL_CNTT": "",
            "FID_INPUT_DATE_1": date,
            "FID_INPUT_HOUR_1": "",
            "FID_RANK_SORT_CLS_CODE": "",
            "FID_INPUT_SRNO": input_srno,
        }
        
        # 재시도 로직
        for retry in range(max_retries):
            try:
                res = requests.get(url, headers=headers, params=params, timeout=30)
                if res.status_code == 200:
                    data = res.json()
                    if data.get('rt_cd') == '0':
                        output = data.get('output', [])
                        if not output:
                            return pd.DataFrame(all_data) if all_data else pd.DataFrame()
                        all_data.extend(output)
                        
                        # 연속 조회 확인
                        if data.get('tr_cd') == 'M':
                            input_srno = output[-1].get('cntt_usiq_srno', '')
                            time.sleep(1.0)  # 페이징 간격
                        else:
                            return pd.DataFrame(all_data) if all_data else pd.DataFrame()
                        break  # 성공하면 재시도 루프 탈출
                    else:
                        return pd.DataFrame(all_data) if all_data else pd.DataFrame()
                else:
                    if retry < max_retries - 1:
                        print(f"⚠️ HTTP {res.status_code}, {retry+1}번째 재시도...")
                        time.sleep(5)  # 재시도 전 대기
                    continue
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"⚠️ 오류 발생, {retry+1}번째 재시도... ({str(e)[:50]})")
                    time.sleep(10)  # 연결 오류 시 더 오래 대기
                else:
                    print(f"❌ 최대 재시도 초과: {date}")
                    return pd.DataFrame(all_data) if all_data else pd.DataFrame()
    
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


def collect_stock_news(stock_code: str, stock_name: str, start_date: str, end_date: str, 
                       output_dir: str, token: str = None) -> pd.DataFrame:
    """단일 종목의 기간별 뉴스 수집"""
    
    if token is None:
        token = get_access_token()
    
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    total_days = (end_dt - start_dt).days + 1
    
    print(f"\n📰 {stock_name}({stock_code}) 뉴스 수집")
    print(f"   기간: {start_date} ~ {end_date} ({total_days}일)")
    
    all_news = []
    current_dt = end_dt  # 최신 날짜부터 역순으로
    
    while current_dt >= start_dt:
        date_str = current_dt.strftime("%Y%m%d")
        
        df_day = get_news_by_date(token, stock_code, date_str)
        
        if not df_day.empty:
            df_day['stock_code'] = stock_code
            df_day['stock_name'] = stock_name
            all_news.append(df_day)
        
        current_dt -= timedelta(days=1)
        time.sleep(1.0)  # API 속도 제한
    
    if all_news:
        df_news = pd.concat(all_news, ignore_index=True)
        df_news = df_news.rename(columns=COLUMN_MAPPING)
        
        # 중복 제거
        if '내용_조회용_일련번호' in df_news.columns:
            df_news = df_news.drop_duplicates(subset=['내용_조회용_일련번호'])
        
        # 저장
        output_path = f"{output_dir}/news_{stock_code}_{start_date}_{end_date}.csv"
        df_news.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ {len(df_news)}건 저장: {output_path}")
        
        return df_news
    else:
        print(f"   ⚠️ 뉴스 없음")
        return pd.DataFrame()


def collect_all_stocks_news(start_date: str, end_date: str, output_dir: str = "_data/news"):
    """30개 종목 전체 뉴스 수집"""
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 종목 리스트 로드
    stocks_df = pd.read_csv("D:/stock/target_stocks.csv")
    
    print("=" * 60)
    print("🚀 30개 종목 뉴스 수집 시작")
    print("=" * 60)
    print(f"기간: {start_date} ~ {end_date}")
    print(f"종목 수: {len(stocks_df)}개")
    print(f"저장 위치: {output_dir}")
    print("=" * 60)
    
    # 토큰 발급
    print("\n🔑 토큰 발급 중...")
    token = get_access_token()
    if not token:
        print("❌ 토큰 발급 실패")
        return
    print("✅ 토큰 발급 완료")
    
    # 종목별 수집
    all_results = []
    
    for idx, row in stocks_df.iterrows():
        code = str(row['Code']).zfill(6)
        name = row['Name']
        
        # 이미 수집된 파일이 있으면 건너뛰기
        existing_file = f"{output_dir}/news_{code}_{start_date}_{end_date}.csv"
        if os.path.exists(existing_file):
            print(f"\n[{idx + 1}/{len(stocks_df)}] {name}({code}) - ⏭️ 이미 수집됨, 건너뜀")
            try:
                df_existing = pd.read_csv(existing_file, encoding='utf-8-sig')
                if not df_existing.empty:
                    all_results.append(df_existing)
            except:
                pass
            continue
        
        print(f"\n[{idx + 1}/{len(stocks_df)}] ", end="")
        
        try:
            df_news = collect_stock_news(code, name, start_date, end_date, output_dir, token)
            if not df_news.empty:
                all_results.append(df_news)
        except Exception as e:
            print(f"   ❌ 오류: {e}")
        
        # 종목 간 대기
        time.sleep(2.0)
    
    # 전체 병합
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        merged_path = f"{output_dir}/news_all_{start_date}_{end_date}.csv"
        df_all.to_csv(merged_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "=" * 60)
        print("🎉 수집 완료!")
        print("=" * 60)
        print(f"총 뉴스: {len(df_all):,}건")
        print(f"종목 수: {df_all['stock_code'].nunique()}개")
        print(f"저장: {merged_path}")
        
        return df_all
    
    return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="30개 종목 뉴스 수집")
    parser.add_argument("--start", type=str, default=None, help="시작일 (YYYYMMDD)")
    parser.add_argument("--end", type=str, default=None, help="종료일 (YYYYMMDD)")
    parser.add_argument("--days", type=int, default=365, help="최근 N일 (기본: 365)")
    parser.add_argument("--output", type=str, default="_data/news", help="저장 디렉토리")
    
    args = parser.parse_args()
    
    # 날짜 설정
    if args.end:
        end_date = args.end
    else:
        end_date = datetime.now().strftime("%Y%m%d")
    
    if args.start:
        start_date = args.start
    else:
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y%m%d")
    
    # 수집 실행
    collect_all_stocks_news(start_date, end_date, args.output)

