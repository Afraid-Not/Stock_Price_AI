import requests
import pandas as pd
import time

class KISDataCollector:
    def __init__(self, authenticator):
        self.auth = authenticator
        self.base_url = "https://openapi.koreainvestment.com:9443"

    def fetch_daily_chart(self, code, start_date, end_date):
        path = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        headers = self.auth.get_headers("FHKST03010100")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": start_date,
            "FID_INPUT_DATE_2": end_date,
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "0"
        }
        res = requests.get(f"{self.base_url}{path}", headers=headers, params=params)
        return pd.DataFrame(res.json()['output2']) if res.status_code == 200 and res.json()['rt_cd'] == '0' else pd.DataFrame()

    def fetch_investor_data(self, code, end_date):
        path = "/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
        headers = self.auth.get_headers("FHPTJ04160001")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": end_date,
            "FID_ORG_ADJ_PRC": "0",
            "FID_ETC_CLS_CODE": ""
        }
        res = requests.get(f"{self.base_url}{path}", headers=headers, params=params)
        return pd.DataFrame(res.json()['output2']) if res.status_code == 200 and res.json()['rt_cd'] == '0' else pd.DataFrame()

    def collect_full_range(self, code, start_date, end_date):
        # 일봉 수집 (100일 단위 루프)
        all_charts = []
        curr_dt = pd.to_datetime(end_date)
        start_dt = pd.to_datetime(start_date)

        while curr_dt >= start_dt:
            req_start = max(start_dt, curr_dt - pd.Timedelta(days=99))
            df_temp = self.fetch_daily_chart(code, req_start.strftime("%Y%m%d"), curr_dt.strftime("%Y%m%d"))
            if not df_temp.empty: all_charts.append(df_temp)
            curr_dt = req_start - pd.Timedelta(days=1)
            time.sleep(0.1)

        df_chart = pd.concat(all_charts).drop_duplicates('stck_bsop_date').sort_values('stck_bsop_date')

        # 투자자 수집 (단순화된 루프)
        all_investors = []
        curr_inv_date = end_date
        while True:
            df_inv = self.fetch_investor_data(code, curr_inv_date)
            if df_inv.empty: break
            all_investors.append(df_inv)
            min_date = df_inv['stck_bsop_date'].min()
            if min_date <= start_date: break
            curr_inv_date = (pd.to_datetime(min_date) - pd.Timedelta(days=1)).strftime("%Y%m%d")
            time.sleep(0.1)

        df_investor = pd.concat(all_investors).drop_duplicates('stck_bsop_date')
        return pd.merge(df_chart, df_investor, on='stck_bsop_date', how='left', suffixes=('', '_inv'))