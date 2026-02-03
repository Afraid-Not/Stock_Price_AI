"""
한국투자증권 API를 사용하여 뉴스 데이터를 수집하는 모듈
"""
import requests
import json
import pandas as pd
import time
import os
import sys
import argparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 상위 디렉토리의 s00_get_token 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '_v6'))
from s00_get_token import get_access_token

load_dotenv()

# 설정
APP_KEY = os.getenv("REAL_APP_KEY")
APP_SECRET = os.getenv("REAL_APP_SECRET")
BASE_URL = "https://openapi.koreainvestment.com:9443"

# TR_ID
TR_ID_NEWS_TITLE = "FHKST01011800"  # 종합 시황/공시(제목)

# 재시도 설정
MAX_RETRIES = 5
RETRY_DELAY = 2  # 초

# 컬럼 매핑 (한글명)
COLUMN_MAPPING = {
    'output1': '응답상세',
    'cntt_usiq_srno': '내용_조회용_일련번호',
    'news_ofer_entp_code': '뉴스_제공_업체_코드',
    'data_dt': '작성일자',
    'data_tm': '작성시간',
    'hts_pbnt_titl_cntt': 'HTS_공시_제목_내용',
    'news_lrdv_code': '뉴스_대구분',
    'dorg': '자료원',
    'iscd1': '종목코드1',
    'iscd2': '종목코드2',
    'iscd3': '종목코드3',
    'iscd4': '종목코드4',
    'iscd5': '종목코드5'
}


def get_news_title(
    token: str,
    fid_news_ofer_entp_code: str = "",  # 뉴스 제공 업체 코드 (공백: 전체)
    fid_cond_mrkt_cls_code: str = "",  # 조건 시장 구분 코드 (공백: 전체)
    fid_input_iscd: str = "",  # 입력 종목코드 (공백: 전체, 종목코드: 해당 종목 뉴스)
    fid_titl_cntt: str = "",  # 제목 내용 (공백: 전체, 키워드: 검색)
    fid_input_date_1: str = "",  # 입력 날짜 (공백: 현재기준, YYYYMMDD 형식)
    fid_input_hour_1: str = "",  # 입력 시간 (공백: 현재기준, HHMMSS 형식)
    fid_rank_sort_cls_code: str = "",  # 순위 정렬 구분 코드 (공백: 기본)
    fid_input_srno: str = "",  # 입력 일련번호 (공백: 처음부터)
    tr_cont: str = "",  # 연속 거래 여부
    dataframe: pd.DataFrame = None,  # 누적 데이터프레임
    max_depth: int = 10  # 최대 페이징 깊이
) -> pd.DataFrame:
    """
    종합 시황/공시(제목) API를 호출하여 뉴스 제목 데이터를 가져옵니다.
    
    Args:
        token: 액세스 토큰
        fid_news_ofer_entp_code: 뉴스 제공 업체 코드
        fid_cond_mrkt_cls_code: 조건 시장 구분 코드
        fid_input_iscd: 입력 종목코드 (예: "005930" - 삼성전자)
        fid_titl_cntt: 제목 내용 (키워드 검색)
        fid_input_date_1: 입력 날짜 (YYYYMMDD 형식, 공백: 현재기준)
        fid_input_hour_1: 입력 시간 (HHMMSS 형식, 공백: 현재기준)
        fid_rank_sort_cls_code: 순위 정렬 구분 코드
        fid_input_srno: 입력 일련번호
        tr_cont: 연속 거래 여부
        dataframe: 누적 데이터프레임
        max_depth: 최대 페이징 깊이
        
    Returns:
        DataFrame: 뉴스 제목 데이터
    """
    path = "/uapi/domestic-stock/v1/quotations/news-title"
    url = f"{BASE_URL}{path}"
    
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {token}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": TR_ID_NEWS_TITLE
    }
    
    params = {
        "FID_NEWS_OFER_ENTP_CODE": fid_news_ofer_entp_code,
        "FID_COND_MRKT_CLS_CODE": fid_cond_mrkt_cls_code,
        "FID_INPUT_ISCD": fid_input_iscd,
        "FID_TITL_CNTT": fid_titl_cntt,
        "FID_INPUT_DATE_1": fid_input_date_1,
        "FID_INPUT_HOUR_1": fid_input_hour_1,
        "FID_RANK_SORT_CLS_CODE": fid_rank_sort_cls_code,
        "FID_INPUT_SRNO": fid_input_srno,
    }
    
    # 페이징 깊이 체크
    depth = len(str(dataframe).split('\n')) if dataframe is not None and not dataframe.empty else 0
    if depth >= max_depth:
        print(f"⚠️ 최대 페이징 깊이({max_depth})에 도달했습니다.")
        return dataframe if dataframe is not None else pd.DataFrame()
    
    # 재시도 로직
    for attempt in range(MAX_RETRIES):
        try:
            res = requests.get(url, headers=headers, params=params, timeout=30)
            
            if res.status_code == 200:
                data = res.json()
                
                if data.get('rt_cd') == '0':
                    # 응답 데이터 처리
                    output_data = data.get('output', [])
                    if not isinstance(output_data, list):
                        output_data = [output_data] if output_data else []
                    
                    if output_data:
                        current_data = pd.DataFrame(output_data)
                    else:
                        current_data = pd.DataFrame()
                    
                    # 데이터프레임 병합
                    if dataframe is not None and not dataframe.empty:
                        dataframe = pd.concat([dataframe, current_data], ignore_index=True)
                    else:
                        dataframe = current_data
                    
                    # 연속 거래 여부 확인 (페이징)
                    tr_cont = data.get('tr_cd', '')
                    if tr_cont == "M" and not current_data.empty:
                        print(f"  📄 다음 페이지 조회 중... (깊이: {depth + 1})")
                        time.sleep(1.0)  # API 호출 제한 고려
                        
                        # 마지막 일련번호를 다음 요청에 사용
                        last_srno = current_data['cntt_usiq_srno'].iloc[-1] if 'cntt_usiq_srno' in current_data.columns else ""
                        
                        return get_news_title(
                            token=token,
                            fid_news_ofer_entp_code=fid_news_ofer_entp_code,
                            fid_cond_mrkt_cls_code=fid_cond_mrkt_cls_code,
                            fid_input_iscd=fid_input_iscd,
                            fid_titl_cntt=fid_titl_cntt,
                            fid_input_date_1=fid_input_date_1,
                            fid_input_hour_1=fid_input_hour_1,
                            fid_rank_sort_cls_code=fid_rank_sort_cls_code,
                            fid_input_srno=last_srno,
                            tr_cont="N",
                            dataframe=dataframe,
                            max_depth=max_depth
                        )
                    else:
                        print(f"✅ 데이터 수집 완료. 총 {len(dataframe) if dataframe is not None else 0}건")
                        return dataframe if dataframe is not None else pd.DataFrame()
                else:
                    error_msg = data.get('msg1', '알 수 없는 오류')
                    print(f"⚠️ API 오류 코드: {data.get('rt_cd')}, 메시지: {error_msg}")
                    return dataframe if dataframe is not None else pd.DataFrame()
            else:
                print(f"⚠️ HTTP 상태 코드: {res.status_code}")
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"🔄 연결 오류 발생, {wait_time}초 후 재시도 ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait_time)
            else:
                print(f"❌ 최대 재시도 횟수 초과: {e}")
                return dataframe if dataframe is not None else pd.DataFrame()
    
    return dataframe if dataframe is not None else pd.DataFrame()


def generate_file_path(output_path: str, stock_code: str = "", start_date: str = "", end_date: str = "") -> str:
    """
    출력 경로를 생성합니다. 디렉토리만 입력되면 자동으로 파일명을 생성합니다.
    
    Args:
        output_path: 사용자가 입력한 출력 경로
        stock_code: 종목코드
        start_date: 시작 날짜
        end_date: 종료 날짜
        
    Returns:
        str: 완전한 파일 경로
    """
    # 이미 파일명이 포함되어 있는지 확인 (.csv, .xlsx 등 확장자 확인)
    if output_path.endswith(('.csv', '.xlsx', '.xls', '.json', '.parquet')):
        return output_path
    
    # 디렉토리 경로인 경우
    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # 파일명 생성
    filename_parts = []
    
    if stock_code:
        filename_parts.append(f"news_{stock_code}")
    else:
        filename_parts.append("news_total")
    
    if start_date and end_date:
        if start_date == end_date:
            filename_parts.append(start_date)
        else:
            filename_parts.append(f"{start_date}_{end_date}")
    elif start_date:
        filename_parts.append(start_date)
    else:
        filename_parts.append(datetime.now().strftime("%Y%m%d"))
    
    filename = "_".join(filename_parts) + ".csv"
    
    return os.path.join(output_path, filename)


def collect_news(
    stock_code: str = "",
    keyword: str = "",
    start_date: str = "",
    end_date: str = "",
    save_path: str = None
) -> pd.DataFrame:
    """
    뉴스 데이터를 수집하는 메인 함수
    
    Args:
        stock_code: 종목코드 (예: "005930" - 삼성전자, 공백: 전체)
        keyword: 제목 키워드 검색 (공백: 전체)
        start_date: 시작 날짜 (YYYYMMDD 형식, 공백: 현재기준)
        end_date: 종료 날짜 (YYYYMMDD 형식, 공백: 현재기준)
        save_path: 저장 경로 (None이면 저장하지 않음)
        
    Returns:
        DataFrame: 수집된 뉴스 데이터
        
    Note:
        - start_date와 end_date가 모두 지정되면 기간 범위로 수집합니다
        - API는 날짜별로 조회하므로, 기간 범위는 각 날짜를 순회하며 수집합니다
        - 일반적으로 최근 1년 정도의 데이터를 제공하지만, 정확한 기간 제한은 API에 따라 다를 수 있습니다
    """
    # 토큰 발급
    print("🔑 토큰 발급 중...")
    token = get_access_token()
    if not token:
        print("❌ 토큰 발급 실패")
        return pd.DataFrame()
    print("✅ 토큰 발급 완료")
    
    # 날짜 설정
    if not start_date:
        # 기본값: 오늘 날짜
        start_date = datetime.now().strftime("%Y%m%d")
    
    # 기간 범위 조회인지 확인
    if end_date and start_date != end_date:
        return collect_news_by_period(
            token=token,
            stock_code=stock_code,
            keyword=keyword,
            start_date=start_date,
            end_date=end_date,
            save_path=save_path
        )
    
    print(f"\n📰 뉴스 수집 시작...")
    print(f"  종목코드: {stock_code if stock_code else '전체'}")
    print(f"  키워드: {keyword if keyword else '전체'}")
    print(f"  조회 날짜: {start_date}")
    
    # 뉴스 데이터 수집
    df_news = get_news_title(
        token=token,
        fid_input_iscd=stock_code,
        fid_titl_cntt=keyword,
        fid_input_date_1=start_date,
        fid_input_hour_1="",  # 시간은 공백으로 전체 조회
        max_depth=20  # 최대 20페이지까지 조회
    )
    
    if df_news.empty:
        print("⚠️ 조회된 뉴스가 없습니다.")
        return pd.DataFrame()
    
    # 컬럼명 한글 변환
    df_news = df_news.rename(columns=COLUMN_MAPPING)
    
    print(f"\n✅ 총 {len(df_news)}건의 뉴스 수집 완료")
    
    # 저장
    if save_path:
        df_news.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"💾 저장 완료: {save_path}")
    
    return df_news


def collect_news_by_period(
    token: str,
    stock_code: str = "",
    keyword: str = "",
    start_date: str = "",
    end_date: str = "",
    save_path: str = None
) -> pd.DataFrame:
    """
    기간 범위로 뉴스 데이터를 수집하는 함수
    
    Args:
        token: 액세스 토큰
        stock_code: 종목코드 (예: "005930" - 삼성전자, 공백: 전체)
        keyword: 제목 키워드 검색 (공백: 전체)
        start_date: 시작 날짜 (YYYYMMDD 형식)
        end_date: 종료 날짜 (YYYYMMDD 형식)
        save_path: 저장 경로 (None이면 저장하지 않음)
        
    Returns:
        DataFrame: 수집된 뉴스 데이터
        
    Note:
        - API는 날짜별로 조회하므로, 각 날짜를 순회하며 수집합니다
        - 주말/공휴일 등 장이 열리지 않은 날은 데이터가 없을 수 있습니다
        - 일반적으로 최근 1년 정도의 데이터를 제공하지만, 정확한 기간 제한은 API에 따라 다를 수 있습니다
    """
    print(f"\n📰 기간 범위 뉴스 수집 시작...")
    print(f"  종목코드: {stock_code if stock_code else '전체'}")
    print(f"  키워드: {keyword if keyword else '전체'}")
    print(f"  기간: {start_date} ~ {end_date}")
    
    # 날짜 범위 생성
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    
    if start_dt > end_dt:
        print("❌ 시작 날짜가 종료 날짜보다 늦습니다.")
        return pd.DataFrame()
    
    # 날짜별로 수집
    df_all_news = []
    current_dt = start_dt
    total_days = (end_dt - start_dt).days + 1
    
    print(f"  총 {total_days}일간의 데이터 수집 예정...\n")
    
    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y%m%d")
        day_num = (current_dt - start_dt).days + 1
        
        print(f"  [{day_num}/{total_days}] {date_str} 수집 중...", end=" ")
        
        df_day = get_news_title(
            token=token,
            fid_input_iscd=stock_code,
            fid_titl_cntt=keyword,
            fid_input_date_1=date_str,
            fid_input_hour_1="",
            max_depth=20
        )
        
        if not df_day.empty:
            df_all_news.append(df_day)
            print(f"✅ {len(df_day)}건")
        else:
            print("⚠️ 데이터 없음")
        
        # 다음 날짜로 이동
        current_dt += timedelta(days=1)
        
        # API 호출 제한 고려 (너무 빠른 요청 방지)
        time.sleep(0.5)
    
    # 모든 데이터 병합
    if df_all_news:
        df_news = pd.concat(df_all_news, ignore_index=True)
        # 컬럼명 한글 변환
        df_news = df_news.rename(columns=COLUMN_MAPPING)
        
        # 중복 제거 (혹시 모를 중복 방지)
        if '내용_조회용_일련번호' in df_news.columns:
            df_news = df_news.drop_duplicates(subset=['내용_조회용_일련번호'], keep='first')
        
        # 날짜순 정렬
        if '작성일자' in df_news.columns and '작성시간' in df_news.columns:
            df_news = df_news.sort_values(['작성일자', '작성시간'], ascending=[False, False])
        
        print(f"\n✅ 총 {len(df_news)}건의 뉴스 수집 완료")
        
        # 저장
        if save_path:
            df_news.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"💾 저장 완료: {save_path}")
        
        return df_news
    else:
        print("\n⚠️ 조회된 뉴스가 없습니다.")
        return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="한국투자증권 API를 사용하여 뉴스 데이터를 수집합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 오늘의 전체 뉴스 수집
  python get_news.py -o news_total.csv
  
  # 특정 종목 뉴스 수집 (오늘)
  python get_news.py -c 005930 -o news_005930.csv
  
  # 키워드 검색
  python get_news.py -k "배당" -o news_dividend.csv
  
  # 기간 범위 뉴스 수집
  python get_news.py -c 005930 -s 20260122 -e 20260129 -o news_period.csv
  
  # 최근 7일간 뉴스 수집
  python get_news.py -c 005930 --days 7 -o news_7days.csv
        """
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="저장할 파일 경로 (예: news_total.csv 또는 news_refiner/news_005930.csv)"
    )
    
    parser.add_argument(
        "-c", "--code",
        type=str,
        default="",
        help="종목코드 (예: 005930 - 삼성전자, 기본값: 전체)"
    )
    
    parser.add_argument(
        "-k", "--keyword",
        type=str,
        default="",
        help="제목 키워드 검색 (기본값: 전체)"
    )
    
    parser.add_argument(
        "-s", "--start-date",
        type=str,
        default="",
        help="시작 날짜 (YYYYMMDD 형식, 기본값: 오늘)"
    )
    
    parser.add_argument(
        "-e", "--end-date",
        type=str,
        default="",
        help="종료 날짜 (YYYYMMDD 형식, 기본값: 시작일과 동일)"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="최근 N일간의 뉴스 수집 (예: --days 7)"
    )
    
    args = parser.parse_args()
    
    # days 옵션이 있으면 start_date와 end_date 자동 설정
    if args.days:
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=args.days - 1)).strftime("%Y%m%d")
    else:
        start_date = args.start_date if args.start_date else ""
        end_date = args.end_date if args.end_date else ""
    
    # 출력 경로 생성 (디렉토리만 입력된 경우 자동으로 파일명 생성)
    save_path = generate_file_path(
        output_path=args.output,
        stock_code=args.code,
        start_date=start_date,
        end_date=end_date
    )
    
    # 뉴스 수집 실행
    print("=" * 60)
    print("뉴스 수집 시작")
    print("=" * 60)
    print(f"저장 경로: {save_path}")
    
    df = collect_news(
        stock_code=args.code,
        keyword=args.keyword,
        start_date=start_date,
        end_date=end_date,
        save_path=save_path
    )
    
    if not df.empty:
        print(f"\n수집된 뉴스 정보:")
        print(f"  총 건수: {len(df)}건")
        if '작성일자' in df.columns:
            print(f"  날짜 범위: {df['작성일자'].min()} ~ {df['작성일자'].max()}")
        print(f"\n처음 5건 미리보기:")
        print(df.head())
    else:
        print("\n⚠️ 수집된 뉴스가 없습니다.")

