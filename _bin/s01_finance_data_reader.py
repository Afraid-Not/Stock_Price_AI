import requests
import json
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# .env 파일의 변수명 확인 (REAL_APP_KEY, REAL_APP_SECRET)
APP_KEY = os.getenv("REAL_APP_KEY")
APP_SECRET = os.getenv("REAL_APP_SECRET")
URL_BASE = "https://openapi.koreainvestment.com:9443"

class KoreaInvestmentAPI:
    def __init__(self, app_key, app_secret, base_url):
        self.app_key = app_key
        self.app_secret = app_secret
        self.base_url = base_url
        self.access_token = None
        
        if self.app_key:
            self.get_access_token()
        else:
            raise ValueError("[ERROR] APP_KEY가 설정되지 않았습니다.")

    def get_access_token(self):
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        res = requests.post(url, headers=headers, data=json.dumps(body))
        if res.status_code == 200:
            self.access_token = res.json()['access_token']
            print(f"[INFO] 토큰 발급 성공")
        else:
            raise RuntimeError(f"토큰 발급 실패: {res.text}")

    def get_daily_price(self, code, start_date, end_date):
        """일별 주가 및 PBR 수집 (파라미터 소문자 필수)"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "FHKST03010100"
        }
        
        all_dfs = []
        pbr_value = "0" # PBR 기본값
        current_date = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")

        while current_date <= end_dt:
            next_date = min(current_date + timedelta(days=90), end_dt)
            # [중요] 파라미터 키값은 반드시 소문자여야 합니다.
            params = {
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": code,
                "fid_input_date_1": current_date.strftime("%Y%m%d"),
                "fid_input_date_2": next_date.strftime("%Y%m%d"),
                "fid_period_div_code": "D",
                "fid_org_adj_prc": "1" # 수정주가 사용 권장
            }
            res = requests.get(url, headers=headers, params=params)
            if res.status_code == 200:
                data = res.json()
                # output1에서 PBR 추출 (삼성전자의 경우 1.15 같은 값)
                if 'output1' in data and data['output1']:
                    pbr_value = data['output1'].get('pbr', pbr_value)
                
                if 'output2' in data and data['output2']:
                    temp_df = pd.DataFrame(data['output2'])
                    temp_df['pbr'] = pbr_value # 데이터프레임에 PBR 컬럼 추가
                    all_dfs.append(temp_df)
            
            time.sleep(0.3)
            current_date = next_date + timedelta(days=1)
            
        return pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=['stck_bsop_date']) if all_dfs else pd.DataFrame()

    def get_investor_trend(self, code, start_date, end_date):
        """투자자별 매매동향 수집 (파라미터 소문자 필수)"""
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": "FHPTJ04160001"
        }

        all_dfs = []
        target_date = end_date
        
        while True:
            params = {
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd": code,
                "fid_input_date_1": target_date,
                "fid_org_adj_prc": "1",
                "fid_etc_cls_code": ""
            }
            res = requests.get(url, headers=headers, params=params)
            if res.status_code != 200: break
            
            output = res.json().get('output2', [])
            if not output: break
            
            df = pd.DataFrame(output)
            all_dfs.append(df)
            
            last_date = df['stck_bsop_date'].min()
            if last_date <= start_date: break
            target_date = (datetime.strptime(last_date, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
            time.sleep(0.3)
                
        return pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=['stck_bsop_date']) if all_dfs else pd.DataFrame()

def main():
    api = KoreaInvestmentAPI(APP_KEY, APP_SECRET, URL_BASE)
    targets = pd.read_csv("stockprice/target_stocks.csv", dtype={'Code': str})
    
    START, END = "20220101", "20251231"
    save_dir = "_data/stock"
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, row in targets.iterrows():
        code, name = row['Code'], row['Name']
        print(f"[{idx+1}/{len(targets)}] {name}({code}) 처리 중...")
        
        df_price = api.get_daily_price(code, START, END)
        df_investor = api.get_investor_trend(code, START, END)
        
        # 데이터가 하나라도 없으면 에러 로그 출력 후 스킵
        if df_price.empty or df_investor.empty:
            print(f" -> [경고] 데이터 부족 (주가: {len(df_price)}건, 투자자: {len(df_investor)}건)")
            continue

        # 병합 및 컬럼명 정리
        target_cols = ['prsn_ntby_qty', 'frgn_ntby_qty', 'orgn_ntby_qty']
        merge_cols = ['stck_bsop_date'] + target_cols
        
        try:
            df = pd.merge(df_price, df_investor[merge_cols], on='stck_bsop_date', how='left')
            rename_map = {
                'stck_bsop_date': '일자', 'stck_clpr': '종가', 
                'stck_hgpr': '고가', 'stck_lwpr': '저가',
                'prsn_ntby_qty': '개인순매수', 'frgn_ntby_qty': '외인순매수', 
                'orgn_ntby_qty': '기관순매수', 'pbr': 'PBR'
            }
            df_final = df[list(rename_map.keys())].rename(columns=rename_map)
            
            save_path = f"{save_dir}/stock_{code}_{START}_{END}.csv"
            df_final.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f" -> 성공: {len(df_final)}행 저장 완료")
        except KeyError as e:
            print(f" -> [에러] 필수 컬럼 누락: {e}")
        
        time.sleep(0.5)

if __name__ == "__main__":
    main()