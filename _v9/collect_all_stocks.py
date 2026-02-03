"""
30개 종목 데이터 수집 + 전처리 + 합치기 파이프라인
"""
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from s01_kis_data_get import collect_stock_data
from s02_rename import rename_file
from s03_preprocessing import StockPreprocessor


def get_stock_start_date(code):
    """종목별 데이터 시작일 반환 (stock_date_info.csv 기반)"""
    # 확인된 상장일 정보
    start_dates = {
        '005930': '20100101',  # 삼성전자
        '000660': '20100101',  # SK하이닉스
        '035420': '20100101',  # NAVER
        '035720': '20100101',  # 카카오
        '006400': '20100101',  # 삼성SDI
        '066570': '20100101',  # LG전자
        '034220': '20100101',  # LG디스플레이
        '018260': '20141101',  # 삼성SDS (2014년 상장)
        '030200': '20100101',  # KT
        '017670': '20100101',  # SK텔레콤
        '032640': '20100101',  # LG유플러스
        '259960': '20210810',  # 크래프톤 (2021년 상장)
        '036570': '20100101',  # 엔씨소프트
        '251270': '20170512',  # 넷마블 (2017년 상장)
        '293490': '20200910',  # 카카오게임즈 (2020년 상장)
        '263750': '20170914',  # 펄어비스 (2017년 상장)
        '078340': '20100101',  # 컴투스
        '112040': '20100101',  # 위메이드
        '053800': '20100101',  # 안랩
        '030520': '20100101',  # 한글과컴퓨터
        '012510': '20100101',  # 더존비즈온
        '067160': '20100101',  # SOOP(아프리카TV)
        '032500': '20100101',  # 케이엠더블유
        '218410': '20171220',  # RFHIC (2017년 상장)
        '336370': '20191107',  # 솔루스첨단소재 (2019년 상장)
        '000990': '20100101',  # DB하이텍
        '011070': '20100101',  # LG이노텍
        '353200': '20201215',  # 대덕전자 (2020년 분할상장)
        '222800': '20171025',  # 심텍 (2017년 상장)
        '402340': '20211129',  # SK스퀘어 (2021년 분할상장)
    }
    return start_dates.get(code, '20100101')


def collect_single_stock(code, name, end_date='20260131'):
    """단일 종목 데이터 수집"""
    start_date = get_stock_start_date(code)
    
    print(f"\n{'='*60}")
    print(f"📊 [{name}({code})] 데이터 수집 시작")
    print(f"   기간: {start_date} ~ {end_date}")
    print(f"{'='*60}")
    
    try:
        df = collect_stock_data(code, start_date, end_date)
        if df is not None and not df.empty:
            print(f"✅ [{name}] 수집 완료: {len(df)}건")
            return True
        else:
            print(f"❌ [{name}] 수집 실패 또는 데이터 없음")
            return False
    except Exception as e:
        print(f"❌ [{name}] 수집 중 오류: {e}")
        return False


def preprocess_single_stock(code, name, end_date='20260131'):
    """단일 종목 전처리 (rename + preprocessing)"""
    start_date = get_stock_start_date(code)
    
    # 파일 경로
    raw_file = f"D:/stock/_v9/_data/{code}_{start_date}_{end_date}.csv"
    renamed_file = f"D:/stock/_v9/_data/renamed_{code}_{start_date}_{end_date}.csv"
    preprocessed_file = f"D:/stock/_v9/_data/preprocessed_{code}_{start_date}_{end_date}.csv"
    
    if not os.path.exists(raw_file):
        print(f"⚠️ [{name}] 원본 파일 없음: {raw_file}")
        return False
    
    try:
        # 1. Rename
        print(f"  🔄 [{name}] 컬럼명 변환 중...")
        rename_file(raw_file, renamed_file)
        
        # 2. Preprocessing
        print(f"  🔄 [{name}] 전처리 중...")
        preprocessor = StockPreprocessor(stock_code=code)
        preprocessor.run_pipeline(renamed_file, preprocessed_file)
        
        print(f"✅ [{name}] 전처리 완료")
        return True
        
    except Exception as e:
        print(f"❌ [{name}] 전처리 중 오류: {e}")
        return False


def merge_all_stocks(stocks_df, end_date='20260131'):
    """모든 종목 데이터를 하나로 합치기"""
    print(f"\n{'='*60}")
    print(f"📦 전체 데이터 병합 시작")
    print(f"{'='*60}")
    
    all_data = []
    
    for _, row in stocks_df.iterrows():
        code = str(row['Code']).zfill(6)
        name = row['Name']
        start_date = get_stock_start_date(code)
        
        preprocessed_file = f"D:/stock/_v9/_data/preprocessed_{code}_{start_date}_{end_date}.csv"
        
        if os.path.exists(preprocessed_file):
            df = pd.read_csv(preprocessed_file)
            
            # 종목 정보 추가
            df['stock_code'] = code
            df['stock_name'] = name
            
            all_data.append(df)
            print(f"  ✅ {name}({code}): {len(df)}건 추가")
        else:
            print(f"  ⚠️ {name}({code}): 파일 없음")
    
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # 날짜순 정렬
        merged_df['날짜'] = pd.to_datetime(merged_df['날짜'])
        merged_df = merged_df.sort_values(['날짜', 'stock_code']).reset_index(drop=True)
        
        # 저장
        output_path = f"D:/stock/_v9/_data/merged_all_stocks_{end_date}.csv"
        merged_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n{'='*60}")
        print(f"🎉 병합 완료!")
        print(f"   총 데이터: {len(merged_df):,}건")
        print(f"   종목 수: {merged_df['stock_code'].nunique()}개")
        print(f"   기간: {merged_df['날짜'].min()} ~ {merged_df['날짜'].max()}")
        print(f"   저장 위치: {output_path}")
        print(f"{'='*60}")
        
        return merged_df
    else:
        print("❌ 병합할 데이터가 없습니다.")
        return None


def main(mode='all', end_date='20260131'):
    """
    메인 파이프라인
    
    Args:
        mode: 'collect' (수집만), 'preprocess' (전처리만), 'merge' (병합만), 'all' (전체)
        end_date: 수집 종료일 (YYYYMMDD)
    """
    # 종목 목록 로드
    stocks_df = pd.read_csv("D:/stock/target_stocks.csv")
    stocks_df = stocks_df.dropna(subset=['Code'])
    stocks_df['Code'] = stocks_df['Code'].astype(int).astype(str).str.zfill(6)
    
    print(f"\n{'#'*60}")
    print(f"#  30개 종목 데이터 파이프라인")
    print(f"#  모드: {mode}")
    print(f"#  종료일: {end_date}")
    print(f"#  종목 수: {len(stocks_df)}개")
    print(f"{'#'*60}")
    
    # 1. 데이터 수집
    if mode in ['collect', 'all']:
        print(f"\n\n{'='*60}")
        print(f"📥 [STEP 1] 데이터 수집")
        print(f"{'='*60}")
        
        success_count = 0
        for idx, row in stocks_df.iterrows():
            code = str(row['Code']).zfill(6)
            name = row['Name']
            
            if collect_single_stock(code, name, end_date):
                success_count += 1
            
            # API 호출 제한 고려
            time.sleep(2)
        
        print(f"\n📊 수집 결과: {success_count}/{len(stocks_df)} 종목 성공")
    
    # 2. 전처리
    if mode in ['preprocess', 'all']:
        print(f"\n\n{'='*60}")
        print(f"⚙️ [STEP 2] 데이터 전처리")
        print(f"{'='*60}")
        
        success_count = 0
        for idx, row in stocks_df.iterrows():
            code = str(row['Code']).zfill(6)
            name = row['Name']
            
            if preprocess_single_stock(code, name, end_date):
                success_count += 1
        
        print(f"\n📊 전처리 결과: {success_count}/{len(stocks_df)} 종목 성공")
    
    # 3. 병합
    if mode in ['merge', 'all']:
        merged_df = merge_all_stocks(stocks_df, end_date)
        return merged_df
    
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='30개 종목 데이터 파이프라인')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['collect', 'preprocess', 'merge', 'all'],
                        help='실행 모드: collect(수집), preprocess(전처리), merge(병합), all(전체)')
    parser.add_argument('--end_date', type=str, default='20260131',
                        help='데이터 수집 종료일 (YYYYMMDD)')
    
    args = parser.parse_args()
    
    main(mode=args.mode, end_date=args.end_date)

