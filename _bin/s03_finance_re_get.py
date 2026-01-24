import requests
import json
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class KoreaInvestmentAPI:
    def __init__(self):
        self.app_key = os.getenv("REAL_APP_KEY")
        self.app_secret = os.getenv("REAL_APP_SECRET")
        self.base_url = "https://openapi.koreainvestment.com:9443"
        self.token_file = "token_cache.json"
        self.access_token = self._get_valid_token()

    def _get_valid_token(self):
        if os.path.exists(self.token_file):
            with open(self.token_file, "r") as f:
                try:
                    cache = json.load(f)
                    if time.time() < cache.get("expiry_time", 0) - 600:
                        return cache.get("access_token")
                except: pass
        return self._issue_new_token()

    def _issue_new_token(self):
        url = f"{self.base_url}/oauth2/tokenP"
        body = {"grant_type": "client_credentials", "appkey": self.app_key, "appsecret": self.app_secret}
        res = requests.post(url, headers={"content-type": "application/json"}, data=json.dumps(body))
        if res.status_code == 200:
            data = res.json()
            access_token = data['access_token']
            expiry_time = time.time() + int(data['expires_in'])
            with open(self.token_file, "w") as f:
                json.dump({"access_token": access_token, "expiry_time": expiry_time}, f)
            return access_token
        else:
            raise RuntimeError(f"토큰 발급 실패: {res.text}")

    def get_comprehensive_data(self, code, start_date, end_date):
        """주가 및 수급 통합 수집 로직 (이전과 동일)"""
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        
        # 1. 일별 시세 수집
        price_list = []
        curr = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        
        while curr <= end_dt:
            nxt = min(curr + timedelta(days=90), end_dt)
            params = {
                "fid_cond_mrkt_div_code": "J", "fid_input_iscd": code,
                "fid_input_date_1": curr.strftime("%Y%m%d"),
                "fid_input_date_2": nxt.strftime("%Y%m%d"),
                "fid_period_div_code": "D", "fid_org_adj_prc": "1"
            }
            res = requests.get(f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice", 
                               headers={**headers, "tr_id": "FHKST03010100"}, params=params)
            if res.status_code == 200 and 'output2' in res.json():
                price_list.append(pd.DataFrame(res.json()['output2']))
            curr = nxt + timedelta(days=1)
            time.sleep(0.1)

        if not price_list: return pd.DataFrame()
        df_p = pd.concat(price_list).drop_duplicates('stck_bsop_date')

        # 2. 투자자별 수급 수집
        investor_list = []
        target_date = end_date
        while True:
            params = {"fid_cond_mrkt_div_code": "J", "fid_input_iscd": code,
                      "fid_input_date_1": target_date, "fid_org_adj_prc": "1", "fid_etc_cls_code": ""}
            res = requests.get(f"{self.base_url}/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily",
                               headers={**headers, "tr_id": "FHPTJ04160001"}, params=params)
            output = res.json().get('output2', [])
            if not output: break
            df_i = pd.DataFrame(output)
            investor_list.append(df_i)
            if df_i['stck_bsop_date'].min() <= start_date: break
            target_date = (datetime.strptime(df_i['stck_bsop_date'].min(), "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
            time.sleep(0.1)

        if not investor_list: return df_p
        df_i = pd.concat(investor_list).drop_duplicates('stck_bsop_date')
        
        final = pd.merge(df_p, df_i, on='stck_bsop_date', how='inner', suffixes=('_p', '_i'))
        return self._finalize_df(final)

    def _finalize_df(self, df):
        """
        이동평균(MA) 및 주요 수급 지표를 포함하여 데이터를 최종 정제합니다.
        """
        cols_map = {
            'stck_bsop_date': 'date',
            'stck_clpr_p': 'close',
            'stck_oprc_p': 'open',
            'stck_hgpr_p': 'high',
            'stck_lwpr_p': 'low',
            'acml_vol_p': 'volume',
            'acml_tr_pbmn': 'trading_value',
            'fltt_rt': 'change_rate',
            'prsn_ntby_qty': 'individual',
            'frgn_ntby_qty': 'foreign',
            'orgn_ntby_qty': 'institutional',
            'pgm_ntby_qty': 'program',
            'hts_frgn_ehrt': 'foreign_ratio'
        }
        
        # 1. 컬럼 선택 및 이름 변경
        existing_cols = [c for c in cols_map.keys() if c in df.columns]
        new_df = df[existing_cols].rename(columns=cols_map)
        
        # 2. 숫자형 변환 (NaN 방지를 위해 errors='coerce' 사용)
        numeric_cols = new_df.columns.drop('date')
        new_df[numeric_cols] = new_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        new_df = new_df.sort_values('date').reset_index(drop=True)

        return new_df

# --- 메인 실행부 (인터랙티브) ---
def main():
    api = KoreaInvestmentAPI()
    save_dir = "_data/manual_fetch"
    os.makedirs(save_dir, exist_ok=True)

    print("=== 재현님의 실시간 종목 수집기 ===")
    user_input = input("수집할 종목 코드를 입력하세요 (여러 개일 경우 쉼표로 구분, 예: 005930, 000660): ")
    codes = [c.strip() for c in user_input.split(',')]

    START, END = "20160101", "20251231"

    for code in codes:
        print(f"🔍 {code} 데이터 분석 중...")
        df = api.get_comprehensive_data(code, START, END)
        if not df.empty:
            path = f"{save_dir}/{code}_{START}_{END}.csv"
            df.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"✅ 저장 완료: {path} ({len(df)}행)")
        else:
            print(f"❌ {code} 데이터를 가져오지 못했습니다.")

if __name__ == "__main__":
    main()