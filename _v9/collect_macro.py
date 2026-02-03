"""
매크로 지표 수집: 코스피 지수 + 환율 (USD/KRW)
2010년 ~ 현재까지 일별 데이터
"""
import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv('D:/stock/.env')

from s00_get_token import get_access_token

APP_KEY = os.getenv("REAL_APP_KEY")
APP_SECRET = os.getenv("REAL_APP_SECRET")
BASE_URL = "https://openapi.koreainvestment.com:9443"


def get_kospi_daily(token: str, start_date: str, end_date: str) -> pd.DataFrame:
    """코스피 지수 일별 데이터 조회"""
    path = "/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice"
    url = f"{BASE_URL}{path}"
    
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKUP03500100"
    }
    
    all_data = []
    current_end = end_date
    
    while True:
        params = {
            "FID_COND_MRKT_DIV_CODE": "U",  # 업종
            "FID_INPUT_ISCD": "0001",  # 코스피 지수
            "FID_INPUT_DATE_1": start_date,
            "FID_INPUT_DATE_2": current_end,
            "FID_PERIOD_DIV_CODE": "D"  # 일봉
        }
        
        try:
            res = requests.get(url, headers=headers, params=params, timeout=30)
            if res.status_code == 200:
                data = res.json()
                if data.get('rt_cd') == '0':
                    output2 = data.get('output2', [])
                    if output2:
                        all_data.extend(output2)
                        print(f"   코스피: {len(output2)}건 수집 (~{output2[-1].get('stck_bsop_date', '')})")
                        
                        # 더 이전 데이터가 있으면 계속
                        if len(output2) >= 100:
                            last_date = output2[-1].get('stck_bsop_date', '')
                            if last_date and last_date > start_date:
                                # 하루 전으로 설정
                                current_end = (datetime.strptime(last_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
                                time.sleep(0.5)
                                continue
                        break
                    else:
                        break
                else:
                    print(f"   ⚠️ API 오류: {data.get('msg1')}")
                    break
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            break
    
    if all_data:
        df = pd.DataFrame(all_data)
        return df
    return pd.DataFrame()


def get_kospi_from_yahoo(start_date: str, end_date: str) -> pd.DataFrame:
    """야후 파이낸스에서 코스피 지수 수집"""
    try:
        import yfinance as yf
        
        # 날짜 형식 변환
        start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        
        # 코스피 지수 (^KS11)
        ticker = yf.Ticker("^KS11")
        df = ticker.history(start=start, end=end)
        
        if df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        df['날짜'] = df['Date'].dt.strftime('%Y%m%d')
        df = df.rename(columns={
            'Close': '코스피_종가',
            'Open': '코스피_시가',
            'High': '코스피_고가',
            'Low': '코스피_저가',
            'Volume': '코스피_거래량'
        })
        
        # 피처 계산 (비율)
        df['kospi_return'] = df['코스피_종가'].pct_change()
        df['kospi_gap_ma5'] = df['코스피_종가'] / df['코스피_종가'].rolling(5).mean() - 1
        df['kospi_volatility'] = (df['코스피_고가'] - df['코스피_저가']) / df['코스피_종가']
        
        df = df[['날짜', '코스피_종가', '코스피_시가', '코스피_고가', '코스피_저가',
                 'kospi_return', 'kospi_gap_ma5', 'kospi_volatility']]
        df = df.sort_values('날짜').reset_index(drop=True)
        
        return df
        
    except ImportError:
        print("   ⚠️ yfinance 패키지 필요: pip install yfinance")
        return pd.DataFrame()
    except Exception as e:
        print(f"   ❌ 코스피 수집 오류: {e}")
        return pd.DataFrame()


def get_usdkrw_from_yahoo(start_date: str, end_date: str) -> pd.DataFrame:
    """야후 파이낸스에서 USD/KRW 환율 수집"""
    try:
        import yfinance as yf
        
        # 날짜 형식 변환
        start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
        
        # USD/KRW 환율 (KRW=X)
        ticker = yf.Ticker("KRW=X")
        df = ticker.history(start=start, end=end)
        
        if df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        df['날짜'] = df['Date'].dt.strftime('%Y%m%d')
        df = df.rename(columns={
            'Close': '환율_종가',
            'Open': '환율_시가',
            'High': '환율_고가',
            'Low': '환율_저가'
        })
        
        # 피처 계산 (비율)
        df['usdkrw_return'] = df['환율_종가'].pct_change()
        df['usdkrw_gap_ma5'] = df['환율_종가'] / df['환율_종가'].rolling(5).mean() - 1
        
        df = df[['날짜', '환율_종가', '환율_시가', '환율_고가', '환율_저가', 
                 'usdkrw_return', 'usdkrw_gap_ma5']]
        df = df.sort_values('날짜').reset_index(drop=True)
        
        return df
        
    except ImportError:
        print("   ⚠️ yfinance 패키지 필요: pip install yfinance")
        return pd.DataFrame()
    except Exception as e:
        print(f"   ❌ 환율 수집 오류: {e}")
        return pd.DataFrame()


def get_usdkrw_daily(token: str, start_date: str, end_date: str) -> pd.DataFrame:
    """USD/KRW 환율 일별 데이터 조회"""
    path = "/uapi/overseas-price/v1/quotations/dailyprice"
    url = f"{BASE_URL}{path}"
    
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST03030100"
    }
    
    all_data = []
    current_end = end_date
    
    while True:
        params = {
            "AUTH": "",
            "EXCD": "NAS",  # 나스닥 (환율은 FX로)
            "SYMB": "FX@KRW",  # USD/KRW
            "GUBN": "0",  # 일봉
            "BYMD": current_end,
            "MODP": "1"
        }
        
        try:
            res = requests.get(url, headers=headers, params=params, timeout=30)
            if res.status_code == 200:
                data = res.json()
                if data.get('rt_cd') == '0':
                    output2 = data.get('output2', [])
                    if output2:
                        all_data.extend(output2)
                        print(f"   환율: {len(output2)}건 수집")
                        break
                    else:
                        break
                else:
                    print(f"   ⚠️ 환율 API: {data.get('msg1')}")
                    break
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            break
    
    if all_data:
        df = pd.DataFrame(all_data)
        return df
    return pd.DataFrame()


def collect_macro_data(start_date: str = "20100101", end_date: str = None,
                       output_dir: str = "_data"):
    """매크로 데이터 수집"""
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 매크로 지표 수집")
    print("=" * 60)
    print(f"기간: {start_date} ~ {end_date}")
    print("=" * 60)
    
    # 토큰 발급
    print("\n🔑 토큰 발급 중...")
    token = get_access_token()
    if not token:
        print("❌ 토큰 발급 실패")
        return
    print("✅ 토큰 발급 완료")
    
    # 1. 코스피 지수 수집 (야후 파이낸스 - 전체 기간)
    print("\n📈 코스피 지수 수집 중 (야후 파이낸스)...")
    df_kospi = get_kospi_from_yahoo(start_date, end_date)
    
    if not df_kospi.empty:
        # 컬럼 정리
        df_kospi = df_kospi.rename(columns={
            'stck_bsop_date': '날짜',
            'bstp_nmix_prpr': '코스피_종가',
            'bstp_nmix_oprc': '코스피_시가',
            'bstp_nmix_hgpr': '코스피_고가',
            'bstp_nmix_lwpr': '코스피_저가',
            'acml_vol': '코스피_거래량',
            'prdy_vrss': '코스피_전일대비',
            'prdy_ctrt': '코스피_등락률'
        })
        
        # 숫자 변환
        for col in ['코스피_종가', '코스피_시가', '코스피_고가', '코스피_저가', '코스피_등락률']:
            if col in df_kospi.columns:
                df_kospi[col] = pd.to_numeric(df_kospi[col], errors='coerce')
        
        # 정렬
        df_kospi = df_kospi.sort_values('날짜').reset_index(drop=True)
        
        # 피처 계산 (비율)
        df_kospi['kospi_return'] = df_kospi['코스피_종가'].pct_change()
        df_kospi['kospi_gap_ma5'] = df_kospi['코스피_종가'] / df_kospi['코스피_종가'].rolling(5).mean() - 1
        df_kospi['kospi_volatility'] = (df_kospi['코스피_고가'] - df_kospi['코스피_저가']) / df_kospi['코스피_종가']
        
        kospi_path = f"{output_dir}/kospi_daily.csv"
        df_kospi.to_csv(kospi_path, index=False, encoding='utf-8-sig')
        print(f"✅ 코스피 저장: {kospi_path} ({len(df_kospi)}건)")
        print(f"   기간: {df_kospi['날짜'].min()} ~ {df_kospi['날짜'].max()}")
    else:
        print("⚠️ 코스피 데이터 없음")
        df_kospi = None
    
    time.sleep(1)
    
    # 2. 환율 수집 (야후 파이낸스)
    print("\n💱 USD/KRW 환율 수집 중 (야후 파이낸스)...")
    df_usdkrw = get_usdkrw_from_yahoo(start_date, end_date)
    
    if df_usdkrw is not None and not df_usdkrw.empty:
        usdkrw_path = f"{output_dir}/usdkrw_daily.csv"
        df_usdkrw.to_csv(usdkrw_path, index=False, encoding='utf-8-sig')
        print(f"✅ 환율 저장: {usdkrw_path} ({len(df_usdkrw)}건)")
        print(f"   기간: {df_usdkrw['날짜'].min()} ~ {df_usdkrw['날짜'].max()}")
    else:
        print("⚠️ 환율 데이터 없음 (yfinance 설치 필요: pip install yfinance)")
    
    # 3. 기존 데이터와 병합
    stock_data_path = f"{output_dir}/merged_all_stocks_20260131.csv"
    if os.path.exists(stock_data_path):
        print(f"\n📦 기존 데이터와 병합 중...")
        df_stock = pd.read_csv(stock_data_path)
        
        # 날짜 형식 통일
        df_stock['날짜'] = pd.to_datetime(df_stock['날짜']).dt.strftime('%Y%m%d')
        
        original_cols = len(df_stock.columns)
        original_rows = len(df_stock)
        
        # 코스피 병합
        if df_kospi is not None and not df_kospi.empty:
            kospi_cols = ['날짜', 'kospi_return', 'kospi_gap_ma5', 'kospi_volatility']
            kospi_cols = [c for c in kospi_cols if c in df_kospi.columns]
            df_stock = df_stock.merge(df_kospi[kospi_cols], on='날짜', how='left')
            print(f"   ✅ 코스피 피처 추가")
        
        # 환율 병합
        if df_usdkrw is not None and not df_usdkrw.empty:
            usdkrw_cols = ['날짜', 'usdkrw_return', 'usdkrw_gap_ma5']
            usdkrw_cols = [c for c in usdkrw_cols if c in df_usdkrw.columns]
            df_stock = df_stock.merge(df_usdkrw[usdkrw_cols], on='날짜', how='left')
            print(f"   ✅ 환율 피처 추가")
        
        # NaN 처리 (매크로 데이터 없는 날은 0으로)
        macro_cols = ['kospi_return', 'kospi_gap_ma5', 'kospi_volatility', 
                      'usdkrw_return', 'usdkrw_gap_ma5']
        for col in macro_cols:
            if col in df_stock.columns:
                df_stock[col] = df_stock[col].fillna(0)
        
        # 저장
        output_path = f"{output_dir}/merged_with_macro.csv"
        df_stock.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        new_cols = len(df_stock.columns)
        print(f"\n💾 병합 완료: {output_path}")
        print(f"   행: {original_rows:,}건")
        print(f"   컬럼: {original_cols} → {new_cols} (+{new_cols - original_cols})")
    else:
        print(f"\n⚠️ 기존 데이터 없음: {stock_data_path}")
    
    print("\n" + "=" * 60)
    print("🎉 수집 및 병합 완료!")
    print("=" * 60)
    
    return df_kospi


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="매크로 지표 수집")
    parser.add_argument("--start", type=str, default="20100101", help="시작일")
    parser.add_argument("--end", type=str, default=None, help="종료일")
    parser.add_argument("--output", type=str, default="_data", help="출력 디렉토리")
    
    args = parser.parse_args()
    
    collect_macro_data(args.start, args.end, args.output)

