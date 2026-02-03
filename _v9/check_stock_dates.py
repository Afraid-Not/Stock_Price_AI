"""
각 종목별 데이터 시작일 확인 스크립트
"""
import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv
from s00_get_token import get_access_token

load_dotenv()

APP_KEY = os.getenv("REAL_APP_KEY")
APP_SECRET = os.getenv("REAL_APP_SECRET")
BASE_URL = "https://openapi.koreainvestment.com:9443"


def get_stock_first_date(token, code):
    """종목의 가장 오래된 데이터 날짜 확인 (2010년부터 조회 시도)"""
    path = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    url = f"{BASE_URL}{path}"
    
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST03010100"
    }
    
    # 2010년 1월부터 100일간 데이터 조회 시도
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": "20100101",
        "FID_INPUT_DATE_2": "20100430",
        "FID_PERIOD_DIV_CODE": "D",
        "FID_ORG_ADJ_PRC": "0"
    }
    
    try:
        res = requests.get(url, headers=headers, params=params, timeout=30)
        if res.status_code == 200:
            data = res.json()
            if data.get('rt_cd') == '0':
                df = pd.DataFrame(data.get('output2', []))
                if not df.empty and 'stck_bsop_date' in df.columns:
                    # 2010년 데이터가 있음
                    return df['stck_bsop_date'].min(), "2010년부터 존재"
                else:
                    # 2010년 데이터 없음 - 최근 데이터로 상장일 추정
                    return None, "2010년 데이터 없음"
    except Exception as e:
        return None, f"오류: {e}"
    
    return None, "조회 실패"


def get_stock_listing_info(token, code):
    """종목의 가장 오래된 데이터 확인 (최근부터 역추적)"""
    path = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    url = f"{BASE_URL}{path}"
    
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST03010100"
    }
    
    # 여러 기간 시도
    test_periods = [
        ("20100101", "20100430"),
        ("20150101", "20150430"),
        ("20180101", "20180430"),
        ("20200101", "20200430"),
        ("20210101", "20210430"),
        ("20220101", "20220430"),
    ]
    
    first_found_year = None
    
    for start, end in test_periods:
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": start,
            "FID_INPUT_DATE_2": end,
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0"
        }
        
        try:
            res = requests.get(url, headers=headers, params=params, timeout=30)
            if res.status_code == 200:
                data = res.json()
                if data.get('rt_cd') == '0':
                    df = pd.DataFrame(data.get('output2', []))
                    if not df.empty and 'stck_bsop_date' in df.columns:
                        if first_found_year is None:
                            first_found_year = start[:4]
                        break
        except:
            pass
        
        time.sleep(0.3)
    
    return first_found_year


def main():
    # 종목 목록 로드
    stocks_df = pd.read_csv("D:/stock/target_stocks.csv")
    
    token = get_access_token()
    if not token:
        print("❌ 토큰 발급 실패")
        return
    
    print("=" * 60)
    print("📊 종목별 데이터 존재 기간 확인")
    print("=" * 60)
    
    results = []
    
    for _, row in stocks_df.iterrows():
        code = str(row['Code']).zfill(6)
        name = row['Name']
        
        # 2010년 데이터 존재 여부 확인
        first_date, status = get_stock_first_date(token, code)
        
        if first_date:
            print(f"✅ {name}({code}): 2010년부터 데이터 존재 (첫 날짜: {first_date})")
            results.append({
                'code': code,
                'name': name,
                'start_year': '2010',
                'has_2010_data': True
            })
        else:
            # 2010년 데이터가 없으면 더 최근 기간 확인
            found_year = get_stock_listing_info(token, code)
            if found_year:
                print(f"⚠️  {name}({code}): {found_year}년부터 데이터 존재")
                results.append({
                    'code': code,
                    'name': name,
                    'start_year': found_year,
                    'has_2010_data': False
                })
            else:
                print(f"❌ {name}({code}): 데이터 확인 실패")
                results.append({
                    'code': code,
                    'name': name,
                    'start_year': 'unknown',
                    'has_2010_data': False
                })
        
        time.sleep(0.5)  # API 호출 제한 고려
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv("D:/stock/_v9/_data/stock_date_info.csv", index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print("📋 요약")
    print("=" * 60)
    print(f"2010년부터 데이터 있는 종목: {results_df['has_2010_data'].sum()}개")
    print(f"2010년 이후 상장 종목: {(~results_df['has_2010_data']).sum()}개")
    print(f"\n결과 저장: D:/stock/_v9/_data/stock_date_info.csv")


if __name__ == "__main__":
    main()

