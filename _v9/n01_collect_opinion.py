"""
30개 종목 투자의견 수집 스크립트
한투 API를 사용하여 증권사별 투자의견/목표가를 수집합니다.

장점:
- 뉴스보다 데이터 양 적음 (빠른 수집)
- 이미 정량화된 데이터 (감성 분석 불필요)
- 전문가 의견이라 신뢰도 높음
"""
import pandas as pd
import requests
import time
import os
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

# 투자의견 TR_ID
TR_ID_OPINION = "FHKST01010600"  # 국내주식 종목투자의견
TR_ID_OPINION_SEC = "FHKST01010700"  # 증권사별 투자의견


def get_invest_opinion(token: str, stock_code: str, debug: bool = False) -> pd.DataFrame:
    """종목 투자의견 조회"""
    path = "/uapi/domestic-stock/v1/quotations/invest-opinion"
    url = f"{BASE_URL}{path}"
    
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": TR_ID_OPINION
    }
    
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",  # 주식
        "FID_INPUT_ISCD": stock_code
    }
    
    try:
        res = requests.get(url, headers=headers, params=params, timeout=30)
        if res.status_code == 200:
            data = res.json()
            
            if debug:
                print(f"   [DEBUG] rt_cd: {data.get('rt_cd')}, msg: {data.get('msg1')}")
                print(f"   [DEBUG] output type: {type(data.get('output'))}")
                if data.get('output'):
                    print(f"   [DEBUG] output: {str(data.get('output'))[:200]}...")
            
            if data.get('rt_cd') == '0':
                output = data.get('output', [])
                if output:
                    # output이 list인 경우
                    if isinstance(output, list):
                        return pd.DataFrame(output)
                    # output이 dict인 경우 (단일 결과)
                    elif isinstance(output, dict):
                        return pd.DataFrame([output])
    except Exception as e:
        print(f"⚠️ API 오류: {e}")
    
    return pd.DataFrame()


def get_invest_opinion_by_sec(token: str, stock_code: str, debug: bool = False) -> pd.DataFrame:
    """증권사별 투자의견 조회"""
    path = "/uapi/domestic-stock/v1/quotations/invest-opbysec"
    url = f"{BASE_URL}{path}"
    
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": TR_ID_OPINION_SEC
    }
    
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": stock_code
    }
    
    try:
        res = requests.get(url, headers=headers, params=params, timeout=30)
        if res.status_code == 200:
            data = res.json()
            
            if debug:
                print(f"   [DEBUG-SEC] rt_cd: {data.get('rt_cd')}, msg: {data.get('msg1')}")
                print(f"   [DEBUG-SEC] output type: {type(data.get('output'))}")
                if data.get('output'):
                    print(f"   [DEBUG-SEC] output: {str(data.get('output'))[:200]}...")
            
            if data.get('rt_cd') == '0':
                output = data.get('output', [])
                if output:
                    if isinstance(output, list):
                        return pd.DataFrame(output)
                    elif isinstance(output, dict):
                        return pd.DataFrame([output])
    except Exception as e:
        print(f"⚠️ API 오류: {e}")
    
    return pd.DataFrame()


def collect_all_opinions(output_dir: str = "_data/opinion"):
    """30개 종목 투자의견 수집"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 종목 리스트 로드
    stocks_df = pd.read_csv("D:/stock/target_stocks.csv")
    
    print("=" * 60)
    print("🚀 30개 종목 투자의견 수집")
    print("=" * 60)
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
    
    # 결과 저장
    all_opinions = []
    all_opinions_sec = []
    
    for idx, row in stocks_df.iterrows():
        code = str(row['Code']).zfill(6)
        name = row['Name']
        
        # 첫 번째 종목은 디버그 모드로 실행
        debug = (idx == 0)
        
        print(f"\n[{idx + 1}/{len(stocks_df)}] {name}({code})")
        
        # 종목 투자의견
        df_opinion = get_invest_opinion(token, code, debug=debug)
        if not df_opinion.empty:
            df_opinion['stock_code'] = code
            df_opinion['stock_name'] = name
            all_opinions.append(df_opinion)
            print(f"   ✅ 투자의견: {len(df_opinion)}건")
            if debug:
                print(f"   [DEBUG] 컬럼: {df_opinion.columns.tolist()}")
        else:
            print(f"   ⚠️ 투자의견 없음")
        
        time.sleep(0.5)
        
        # 증권사별 투자의견
        df_sec = get_invest_opinion_by_sec(token, code, debug=debug)
        if not df_sec.empty:
            df_sec['stock_code'] = code
            df_sec['stock_name'] = name
            all_opinions_sec.append(df_sec)
            print(f"   ✅ 증권사별: {len(df_sec)}건")
            if debug:
                print(f"   [DEBUG-SEC] 컬럼: {df_sec.columns.tolist()}")
        else:
            print(f"   ⚠️ 증권사별 없음")
        
        time.sleep(0.5)
    
    # 저장
    if all_opinions:
        df_all = pd.concat(all_opinions, ignore_index=True)
        path = f"{output_dir}/invest_opinion_all.csv"
        df_all.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"\n💾 투자의견 저장: {path} ({len(df_all)}건)")
    
    if all_opinions_sec:
        df_sec_all = pd.concat(all_opinions_sec, ignore_index=True)
        path_sec = f"{output_dir}/invest_opinion_by_sec.csv"
        df_sec_all.to_csv(path_sec, index=False, encoding='utf-8-sig')
        print(f"💾 증권사별 저장: {path_sec} ({len(df_sec_all)}건)")
    
    print("\n" + "=" * 60)
    print("🎉 수집 완료!")
    print("=" * 60)
    
    return df_all if all_opinions else pd.DataFrame()


def process_opinion_features(opinion_path: str, stock_data_path: str, 
                             output_path: str = None) -> pd.DataFrame:
    """
    투자의견을 피처로 변환하여 주가 데이터와 병합
    
    피처:
    - target_price_ratio: 목표가 / 현재가 - 1 (상승 여력)
    - opinion_score: 투자의견 점수 (매수=1, 중립=0, 매도=-1)
    - opinion_change: 투자의견 변경 여부
    - analyst_count: 최근 리포트 수 (관심도)
    """
    print("=" * 60)
    print("📊 투자의견 피처 생성")
    print("=" * 60)
    
    # 데이터 로드
    print(f"\n📂 투자의견 로드: {opinion_path}")
    df_opinion = pd.read_csv(opinion_path, encoding='utf-8-sig')
    print(f"   {len(df_opinion)}건")
    
    print(f"\n📂 주가 데이터 로드: {stock_data_path}")
    df_stock = pd.read_csv(stock_data_path)
    print(f"   {len(df_stock)}건")
    
    # 컬럼 확인 및 출력
    print(f"\n📋 투자의견 컬럼: {df_opinion.columns.tolist()}")
    
    # 투자의견 피처 생성 (예시 - 실제 컬럼명에 맞게 수정 필요)
    # 한투 API 응답 형태에 따라 달라질 수 있음
    
    # 예시 피처 생성 로직:
    # 1. 목표가 대비 현재가 비율
    # 2. 투자의견 점수화
    # 3. 최근 리포트 수
    
    print("\n⚠️ 실제 API 응답 컬럼을 확인 후 피처 생성 로직 수정 필요")
    print("   df_opinion.head() 결과:")
    print(df_opinion.head())
    
    return df_opinion


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="투자의견 수집")
    parser.add_argument("--output", type=str, default="_data/opinion", help="저장 디렉토리")
    parser.add_argument("--process", action="store_true", help="피처 생성 모드")
    parser.add_argument("--opinion", type=str, help="투자의견 CSV 경로")
    parser.add_argument("--stock", type=str, help="주가 데이터 경로")
    
    args = parser.parse_args()
    
    if args.process:
        if args.opinion and args.stock:
            process_opinion_features(args.opinion, args.stock)
        else:
            print("❌ --opinion과 --stock 경로를 지정해주세요")
    else:
        collect_all_opinions(args.output)

