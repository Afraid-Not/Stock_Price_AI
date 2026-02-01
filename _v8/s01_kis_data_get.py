import requests
import json
import pandas as pd
import time
from s00_get_token import get_access_token
import os
from dotenv import load_dotenv

load_dotenv()

# 설정
APP_KEY = os.getenv("REAL_APP_KEY")
APP_SECRET = os.getenv("REAL_APP_SECRET")
BASE_URL = "https://openapi.koreainvestment.com:9443"

# TR_ID
# 주식 기간별 시세 조회 (실전용)
TR_ID_DAILY_CHART = "FHKST03010100" 
# 종목별 투자자매매동향 (실전용)
TR_ID_INVESTOR = "FHPTJ04160001"

# 재시도 설정
MAX_RETRIES = 5
RETRY_DELAY = 2  # 초

def get_stock_daily_chart(token, code, start_date, end_date):
    """
    주식 일봉 차트 조회 (시가, 고가, 저가, 종가, 거래량 등)
    재시도 로직 포함
    """
    path = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
    url = f"{BASE_URL}{path}"
    
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": TR_ID_DAILY_CHART
    }
    
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",      
        "FID_INPUT_ISCD": code,             
        "FID_INPUT_DATE_1": start_date,     
        "FID_INPUT_DATE_2": end_date,       
        "FID_PERIOD_DIV_CODE": "D",         
        "FID_ORG_ADJ_PRC": "0"              
    }
    
    # 재시도 로직
    for attempt in range(MAX_RETRIES):
        try:
            res = requests.get(url, headers=headers, params=params, timeout=30)
            if res.status_code == 200:
                data = res.json()
                if data.get('rt_cd') == '0':
                    return pd.DataFrame(data.get('output2', []))
                else:
                    print(f"  ⚠️ API 오류 코드: {data.get('rt_cd')}, 메시지: {data.get('msg1', '')}")
                    return pd.DataFrame()
            else:
                print(f"  ⚠️ HTTP 상태 코드: {res.status_code}")
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, 
                requests.exceptions.RequestException) as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"  🔄 연결 오류 발생, {wait_time}초 후 재시도 ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ 최대 재시도 횟수 초과: {e}")
                return pd.DataFrame()
    
    return pd.DataFrame()

def get_investor_daily(token, code, date):
    """
    종목별 투자자 매매동향 (외국인, 기관, 개인 순매수 등)
    재시도 로직 포함
    """
    path = "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
    url = f"{BASE_URL}{path}"
    
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": TR_ID_INVESTOR
    }
    
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": date, # 조회 종료일 (이 날짜부터 과거 데이터 조회)
        "FID_ORG_ADJ_PRC": "0",
        "FID_ETC_CLS_CODE": ""
    }
    
    # 재시도 로직
    for attempt in range(MAX_RETRIES):
        try:
            res = requests.get(url, headers=headers, params=params, timeout=30)
            if res.status_code == 200:
                data = res.json()
                if data.get('rt_cd') == '0':
                    return pd.DataFrame(data.get('output2', []))
                else:
                    print(f"  ⚠️ API 오류 코드: {data.get('rt_cd')}, 메시지: {data.get('msg1', '')}")
                    return pd.DataFrame()
            else:
                print(f"  ⚠️ HTTP 상태 코드: {res.status_code}")
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, 
                requests.exceptions.RequestException) as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"  🔄 연결 오류 발생, {wait_time}초 후 재시도 ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait_time)
            else:
                print(f"  ❌ 최대 재시도 횟수 초과: {e}")
                return pd.DataFrame()
    
    return pd.DataFrame()

def collect_stock_data(code, start_date, end_date):
    token = get_access_token()
    if not token:
        print("토큰 발급 실패")
        return

    print(f"[{code}] 데이터 수집 시작...")

    # 1. 일봉 차트 데이터 수집
    df_chart_list = []
    
    # 100일 단위로 끊어서 요청 (API 제한 고려)
    # 시작일부터 종료일까지 날짜 생성
    start_dt = pd.to_datetime(start_date, format='%Y%m%d')
    end_dt = pd.to_datetime(end_date, format='%Y%m%d')
    
    current_dt = end_dt
    
    print("일봉 차트 데이터 수집 중...")
    
    while current_dt >= start_dt:
        # 100일 전 날짜 계산 (종료일 기준)
        # API가 종료일(FID_INPUT_DATE_2)을 기준으로 과거 데이터를 가져오므로
        # 루프를 돌면서 종료일을 과거로 이동시켜야 함
        # 그러나 이 API는 시작일(1)과 종료일(2)을 지정하면 그 사이 데이터를 줌.
        # 단, 최대 100일까지만 가능.
        
        # 방식 변경: start_dt 부터 100일씩 끊어서 조회하는게 직관적일 수 있으나
        # API 특성상 최신순 정렬이 기본이므로, end_dt부터 역순으로 100일씩 끊어서 조회
        
        # 조회 구간 설정
        # 이번 구간의 종료일: current_dt
        # 이번 구간의 시작일: current_dt - 99일 (총 100일)
        
        req_end_dt = current_dt
        req_start_dt = current_dt - pd.Timedelta(days=99)
        
        # 만약 계산된 시작일이 전체 시작일보다 이전이면 전체 시작일로 조정
        if req_start_dt < start_dt:
            req_start_dt = start_dt
            
        s_date_str = req_start_dt.strftime("%Y%m%d")
        e_date_str = req_end_dt.strftime("%Y%m%d")
        
        print(f"  기간 요청: {s_date_str} ~ {e_date_str}")
        
        df_temp = get_stock_daily_chart(token, code, s_date_str, e_date_str)
        
        if not df_temp.empty:
            df_chart_list.append(df_temp)
        else:
            print("  데이터 없음 또는 수집 실패")
            
        # 다음 구간을 위해 날짜 이동 (시작일 하루 전으로)
        current_dt = req_start_dt - pd.Timedelta(days=1)
        
        # API 호출 제한 고려 (초당 요청 제한 등)
        time.sleep(1.0)  # 대기 시간 증가 
        
    if df_chart_list:
        df_chart = pd.concat(df_chart_list, ignore_index=True)
        # 컬럼 확인
        if df_chart.empty:
            print("⚠️ 차트 데이터가 비어있습니다.")
            df_chart = pd.DataFrame()
        elif 'stck_bsop_date' not in df_chart.columns:
            print(f"⚠️ 차트 데이터에 'stck_bsop_date' 컬럼이 없습니다. 컬럼: {list(df_chart.columns)}")
            # API 응답의 실제 컬럼명 확인 (일반적으로 stck_bsop_date 또는 다른 이름일 수 있음)
            date_cols = [col for col in df_chart.columns if 'date' in col.lower() or '일자' in col or 'bsop' in col.lower()]
            if date_cols:
                print(f"  날짜 관련 컬럼 발견: {date_cols}, 첫 번째 컬럼 사용")
                df_chart = df_chart.rename(columns={date_cols[0]: 'stck_bsop_date'})
            else:
                print("  날짜 컬럼을 찾을 수 없습니다.")
                df_chart = pd.DataFrame()
        else:
            # 중복 제거 (혹시 모를 겹침 방지)
            df_chart.drop_duplicates(subset=['stck_bsop_date'], inplace=True)
            # 날짜 오름차순 정렬
            df_chart.sort_values('stck_bsop_date', inplace=True)
    else:
        df_chart = pd.DataFrame()


    # 2. 투자자 매매동향 데이터 수집 (날짜 기준 내림차순으로 옴)
    # 이 API(investor-trade-by-stock-daily)는 FID_INPUT_DATE_1 (조회기준일자) 기준으로
    # 과거 일정 기간의 데이터를 리스트로 줌 (보통 30~100건 정도)
    # 페이징이 안되거나 기간 지정이 모호할 수 있으므로,
    # 반복 호출하여 데이터를 모아야 함.
    
    print("투자자 매매동향 데이터 수집 중...")
    df_investor_list = []
    
    # 투자자 데이터 수집 전략:
    # 가장 최근 날짜(end_date)부터 시작해서 과거로 가면서 수집
    # API 응답의 가장 마지막 날짜(가장 과거)를 확인하고, 그 전날을 다시 기준일로 요청
    
    curr_date_str = end_date
    target_start_str = start_date
    
    # 무한 루프 방지용 카운터
    max_loops = 1000 
    loop_cnt = 0
    
    last_collected_date = "99999999"
    
    while loop_cnt < max_loops:
        print(f"  기준일 요청: {curr_date_str}")
        df_inv_temp = get_investor_daily(token, code, curr_date_str)
        
        if df_inv_temp.empty:
            print("  데이터 없음")
            break
            
        # 수집된 데이터 중 가장 과거 날짜 확인
        # 날짜 컬럼: stck_bsop_date
        if 'stck_bsop_date' not in df_inv_temp.columns:
            print("  날짜 컬럼 확인 불가")
            break
            
        min_date_in_batch = df_inv_temp['stck_bsop_date'].min()
        max_date_in_batch = df_inv_temp['stck_bsop_date'].max()
        
        # 이번에 가져온 데이터가 이미 수집한 범위 내에 완전히 포함되어 더 이상 새로운게 없으면 중단
        if max_date_in_batch >= last_collected_date:
             # 겹치는 부분이 있지만 새로운 과거 데이터가 있을 수 있으므로 필터링 후 추가
             df_inv_temp = df_inv_temp[df_inv_temp['stck_bsop_date'] < last_collected_date]
             if df_inv_temp.empty:
                 print("  더 이상 새로운 데이터 없음")
                 break

        df_investor_list.append(df_inv_temp)
        last_collected_date = min_date_in_batch
        
        # 목표 시작일보다 더 과거 데이터를 가져왔으면 종료
        if min_date_in_batch <= target_start_str:
            print("  목표 시작일 도달")
            break
            
        # 다음 요청을 위해 날짜 이동 (가장 과거 날짜의 전날)
        # 문자열 날짜 계산
        min_dt = pd.to_datetime(min_date_in_batch, format='%Y%m%d')
        next_req_dt = min_dt - pd.Timedelta(days=1)
        curr_date_str = next_req_dt.strftime("%Y%m%d")
        
        loop_cnt += 1
        time.sleep(1.0)  # 대기 시간 증가

    if df_investor_list:
        df_investor = pd.concat(df_investor_list, ignore_index=True)
        df_investor.drop_duplicates(subset=['stck_bsop_date'], inplace=True)
        # 목표 기간 내 데이터만 필터링
        df_investor = df_investor[(df_investor['stck_bsop_date'] >= start_date) & (df_investor['stck_bsop_date'] <= end_date)]
    else:
        df_investor = pd.DataFrame()

    
    # 데이터 병합 (날짜 기준)
    # 차트 데이터: stck_bsop_date (YYYYMMDD)
    # 투자자 데이터: stck_bsop_date (YYYYMMDD)
    
    # 필요한 컬럼만 선택 및 이름 변경 등 전처리 가능
    # 예: df_investor에서 개인/외국인/기관 순매수만 가져오기
    # frgn_ntby_qty (외국인 순매수량), orgn_ntby_qty (기관 순매수량), prsn_ntby_qty (개인 순매수량) 등
    
    print(f"차트 데이터: {len(df_chart)}건, 투자자 데이터: {len(df_investor)}건 수집됨")
    
    # 데이터 검증
    if df_chart.empty:
        print("⚠️ 차트 데이터가 없어 병합을 건너뜁니다.")
        return pd.DataFrame()
    
    if 'stck_bsop_date' not in df_chart.columns:
        print(f"⚠️ 차트 데이터에 'stck_bsop_date' 컬럼이 없습니다. 컬럼: {list(df_chart.columns)}")
        return pd.DataFrame()
    
    # 병합 (Left Join)
    df_merged = pd.merge(df_chart, df_investor, on='stck_bsop_date', how='left', suffixes=('', '_investor'))
    
    # 결과 저장
    filename = f"D:/stock/_v8/_data/{code}_{start_date}_{end_date}.csv"
    df_merged.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"저장 완료: {filename}")
    
    return df_merged

if __name__ == "__main__":
    # 삼성전자, 2010년 1월 ~ 2025년 12월 데이터
    collect_stock_data("005930", "20251201", "20251231")

