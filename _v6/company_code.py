# 2026년 1월 기준 시가총액 상위 종목 코드 리스트
stock_dict = {
    # --- KOSPI Top 20 ---
    "삼성전자": "005930", #
    "SK하이닉스": "000660", #
    "현대차": "005380", #
    "한화에어로스페이스": "012450",
    "기아": "000270", #
    "두산에너빌리티": "034020", 
    "삼성물산": "028260", #
    "KB금융": "105560", #
    "셀트리온": "068270", #
    "한화오션": "042660", #
    "현대모비스": "012330", #
    "NAVER": "035420", #
    "신한지주": "055550", #
    "한국전력": "015760", #
    "고려아연": "010130", #
    "POSCO홀딩스": "005490", #

    # --- KOSDAQ Top 20 ---
    "에코프로비엠": "247540", #
    "에코프로": "086520", #
    "알테오젠": "196170", 
    "리가켐바이오": "141080", #
    "삼천당제약": "000250", #
    "리노공업": "058470", #
    "펩트론": "087010", #
    "HLB": "028300", #
    "파마리서치": "214450", 
    "셀트리온제약": "068760",
    "클래시스": "214150",
    "JYP Ent.": "035900",
    "SM": "041510",
    "동진쎄미켐": "005290",
    "실리콘투": "257720",
    "ISC": "095340"
}

def batch_create_datasets(start_date, end_date, is_train=True):
    """
    모든 종목에 대해 데이터셋 생성
    
    Args:
        start_date: 시작일 (YYYYMMDD)
        end_date: 종료일 (YYYYMMDD)
        is_train: 스케일러 신규 학습 여부 (기본값: True)
    """
    from create_stock_dataset import run_full_pipeline
    
    codes = list(stock_dict.values())
    total = len(codes)
    success_count = 0
    fail_count = 0
    failed_codes = []
    
    print(f"\n{'='*60}")
    print(f"📊 Batch Dataset Creation Started")
    print(f"{'='*60}")
    print(f"Total Companies: {total}")
    print(f"Date Range: {start_date} ~ {end_date}")
    print(f"Train Mode: {is_train}")
    print(f"{'='*60}\n")
    
    for idx, code in enumerate(codes, 1):
        company_name = [k for k, v in stock_dict.items() if v == code][0]
        print(f"\n{'='*60}")
        print(f"[{idx}/{total}] {company_name} ({code})")
        print(f"{'='*60}")
        
        try:
            run_full_pipeline(code, start_date, end_date, is_train)
            success_count += 1
            print(f"✅ [{idx}/{total}] {company_name} ({code}) 완료")
        except Exception as e:
            fail_count += 1
            failed_codes.append((company_name, code, str(e)))
            print(f"❌ [{idx}/{total}] {company_name} ({code}) 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 최종 결과 요약
    print(f"\n{'='*60}")
    print(f"📈 Batch Processing Summary")
    print(f"{'='*60}")
    print(f"Total:        {total}")
    print(f"Success:      {success_count}")
    print(f"Failed:       {fail_count}")
    
    if failed_codes:
        print(f"\n❌ Failed Companies:")
        for name, code, error in failed_codes:
            print(f"   - {name} ({code}): {error}")
    
    print(f"\n✅ Batch processing completed!")

if __name__ == "__main__":
    import argparse
    from datetime import datetime, timedelta
    
    parser = argparse.ArgumentParser(description="모든 종목 데이터셋 일괄 생성")
    parser.add_argument("--start", type=str, default=None, help="시작일 (YYYYMMDD, 미지정시 1년 전)")
    parser.add_argument("--end", type=str, default=None, help="종료일 (YYYYMMDD, 미지정시 오늘)")
    parser.add_argument("--train", action="store_true", help="스케일러 신규 학습 여부")
    
    args = parser.parse_args()
    
    # 날짜 기본값 설정
    if args.end is None:
        args.end = datetime.now().strftime("%Y%m%d")
    
    if args.start is None:
        end_date = datetime.strptime(args.end, "%Y%m%d")
        start_date = end_date - timedelta(days=365)  # 1년 전
        args.start = start_date.strftime("%Y%m%d")
    
    batch_create_datasets(args.start, args.end, args.train)