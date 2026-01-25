import pandas as pd

# 파일 경로
file_path = "D:/stock/_data/manual_fetch/005930_20100101_20251231.csv"
df = pd.read_csv(file_path)

# 컬럼명 매핑 (최대한 상세하게 작성)
rename_map = {
    # --- [1] 기본 시세 데이터 (일봉 API 유래) ---
    'stck_bsop_date': '날짜',
    'stck_oprc': '시가',
    'stck_hgpr': '고가',
    'stck_lwpr': '저가',
    'stck_clpr': '종가',
    'acml_vol': '거래량',
    'acml_tr_pbmn': '거래대금',
    'prdy_vrss': '전일대비',
    'prdy_vrss_sign': '대비부호', # 1:상한, 2:상승, 3:보합, 4:하한, 5:하락
    'prdy_ctrt': '등락률',
    'flng_cls_code': '락구분',    # 00:보통, 01:권리락, 02:배당락...
    'mod_yn': '수정주가여부',
    'revl_issu_reas': '재평가사유',

    # --- [2] 투자자별 순매수 데이터 (투자자 API 유래) ---
    # 수량 (Quantity)
    'prsn_ntby_qty': '개인_순매수수량',
    'frgn_ntby_qty': '외국인_순매수수량',
    'orgn_ntby_qty': '기관계_순매수수량',
    'scrt_ntby_qty': '금융투자_순매수수량',
    'ivtr_ntby_qty': '투신_순매수수량',
    'pe_fund_ntby_vol': '사모펀드_순매수수량', # API 필드명 차이 주의 (vol/qty 혼용)
    'bank_ntby_qty': '은행_순매수수량',
    'insu_ntby_qty': '보험_순매수수량',
    'mrbn_ntby_qty': '기타금융_순매수수량',
    'fund_ntby_qty': '연기금_순매수수량',
    'etc_ntby_qty': '기타법인_순매수수량',
    'etc_corp_ntby_vol': '기타법인_순매수수량_2', # 중복 가능성
    'etc_orgt_ntby_vol': '기타단체_순매수수량',
    'frgn_reg_ntby_qty': '외국인등록_순매수수량',
    'frgn_nreg_ntby_qty': '외국인미등록_순매수수량',

    # 금액 (Price/Amount)
    'prsn_ntby_tr_pbmn': '개인_순매수금액',
    'frgn_ntby_tr_pbmn': '외국인_순매수금액',
    'orgn_ntby_tr_pbmn': '기관계_순매수금액',
    'scrt_ntby_tr_pbmn': '금융투자_순매수금액',
    'ivtr_ntby_tr_pbmn': '투신_순매수금액',
    'pe_fund_ntby_tr_pbmn': '사모펀드_순매수금액',
    'bank_ntby_tr_pbmn': '은행_순매수금액',
    'insu_ntby_tr_pbmn': '보험_순매수금액',
    'mrbn_ntby_tr_pbmn': '기타금융_순매수금액',
    'fund_ntby_tr_pbmn': '연기금_순매수금액',
    'etc_ntby_tr_pbmn': '기타법인_순매수금액',
    'etc_corp_ntby_tr_pbmn': '기타법인_순매수금액_2',
    'etc_orgt_ntby_tr_pbmn': '기타단체_순매수금액',
    'frgn_reg_ntby_pbmn': '외국인등록_순매수금액',
    'frgn_nreg_ntby_pbmn': '외국인미등록_순매수금액',

    # 매수/매도 분할 데이터 (일부 컬럼 존재 시)
    'prsn_seln_vol': '개인_매도수량',
    'prsn_shnu_vol': '개인_매수수량',
    'frgn_seln_vol': '외국인_매도수량',
    'frgn_shnu_vol': '외국인_매수수량',
    'orgn_seln_vol': '기관계_매도수량',
    'orgn_shnu_vol': '기관계_매수수량',
    
    # --- [3] 중복 컬럼 정리 (Merge 과정에서 발생) ---
    # _investor 붙은 컬럼들은 보통 투자자 API에서 온 시세 정보로, 기본 시세와 중복됨
    'stck_clpr_investor': '종가_확인용',
    'stck_oprc_investor': '시가_확인용', 
    'stck_hgpr_investor': '고가_확인용',
    'stck_lwpr_investor': '저가_확인용',
    'acml_vol_investor': '거래량_확인용',
    'acml_tr_pbmn_investor': '거래대금_확인용',
    'prdy_vrss_investor': '전일대비_확인용',
    'prdy_vrss_sign_investor': '대비부호_확인용',
    
    # 기타 확인된 컬럼
    'bold_yn': '볼드여부'
}

# 1. 컬럼명 변경 (매핑에 없는 컬럼은 원래 영문 이름 유지)
df_renamed = df.rename(columns=rename_map)

# 2. 날짜 포맷 변경
# 날짜 컬럼이 '날짜'로 바뀌었으면 변환 수행
if '날짜' in df_renamed.columns:
    df_renamed['날짜'] = pd.to_datetime(df_renamed['날짜'], format='%Y%m%d', errors='coerce')

# 3. 불필요한 중복 컬럼(_확인용) 제거하고 싶으시면 아래 주석 해제
# drop_cols = [c for c in df_renamed.columns if '_확인용' in c]
# df_renamed.drop(columns=drop_cols, inplace=True)

print(f"전체 데이터 크기: {df_renamed.shape}")
print("변경된 컬럼 목록:")
print(df_renamed.columns.tolist())

# 저장
save_path = "D:/stock/_data/manual_fetch/005930_renamed.csv"
df_renamed.to_csv(save_path, index=False, encoding='utf-8-sig')
print(f"전체 컬럼 저장 완료: {save_path}")
