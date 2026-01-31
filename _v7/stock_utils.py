import pandas as pd

class StockRenamer:
    # 모든 컬럼 매핑 정보를 클래스 변수로 관리
    RENAME_MAP = {
        # [기본 시세]
        'stck_bsop_date': '날짜', 'stck_clpr': '종가', 'stck_oprc': '시가',
        'stck_hgpr': '고가', 'stck_lwpr': '저가', 'acml_vol': '거래량',
        'acml_tr_pbmn': '거래대금', 'prdy_vrss': '전일대비', 'prdy_vrss_sign': '대비부호',
        'prdy_ctrt': '등락률', 'flng_cls_code': '락구분', 'mod_yn': '수정주가여부',
        'revl_issu_reas': '재평가사유', 'prtt_rate': '분할비율',

        # [투자자 순매수]
        'prsn_ntby_qty': '개인_순매수수량', 'frgn_ntby_qty': '외국인_순매수수량',
        'orgn_ntby_qty': '기관계_순매수수량', 'scrt_ntby_qty': '금융투자_순매수수량',
        'ivtr_ntby_qty': '투신_순매수수량', 'pe_fund_ntby_vol': '사모펀드_순매수수량',
        'bank_ntby_qty': '은행_순매수수량', 'insu_ntby_qty': '보험_순매수수량',
        'mrbn_ntby_qty': '기타금융_순매수수량', 'fund_ntby_qty': '연기금_순매수수량',
        'etc_ntby_qty': '기타법인_순매수수량', 'etc_corp_ntby_vol': '기타법인_순매수수량_2',
        'etc_orgt_ntby_vol': '기타단체_순매수수량', 'frgn_reg_ntby_qty': '외국인등록_순매수수량',
        'frgn_nreg_ntby_qty': '외국인미등록_순매수수량',
        'prsn_ntby_tr_pbmn': '개인_순매수금액', 'frgn_ntby_tr_pbmn': '외국인_순매수금액',
        'orgn_ntby_tr_pbmn': '기관계_순매수금액', 'scrt_ntby_tr_pbmn': '금융투자_순매수금액',
        'ivtr_ntby_tr_pbmn': '투신_순매수금액', 'pe_fund_ntby_tr_pbmn': '사모펀드_순매수금액',
        'bank_ntby_tr_pbmn': '은행_순매수금액', 'insu_ntby_tr_pbmn': '보험_순매수금액',
        'mrbn_ntby_tr_pbmn': '기타금융_순매수금액', 'fund_ntby_tr_pbmn': '연기금_순매수금액',
        'etc_ntby_tr_pbmn': '기타법인_순매수금액', 'etc_corp_ntby_tr_pbmn': '기타법인_순매수금액_2',
        'etc_orgt_ntby_tr_pbmn': '기타단체_순매수금액', 'frgn_reg_ntby_pbmn': '외국인등록_순매수금액',
        'frgn_nreg_ntby_pbmn': '외국인미등록_순매수금액',

        # [세부 매도/매수 수량]
        'prsn_seln_vol': '개인_매도수량', 'prsn_shnu_vol': '개인_매수수량',
        'frgn_seln_vol': '외국인_매도수량', 'frgn_shnu_vol': '외국인_매수수량',
        'orgn_seln_vol': '기관계_매도수량', 'orgn_shnu_vol': '기관계_매수수량',
        'scrt_seln_vol': '금융투자_매도수량', 'scrt_shnu_vol': '금융투자_매수수량',
        'ivtr_seln_vol': '투신_매도수량', 'ivtr_shnu_vol': '투신_매수수량',
        'pe_fund_seln_vol': '사모펀드_매도수량', 'pe_fund_shnu_vol': '사모펀드_매수수량',
        'bank_seln_vol': '은행_매도수량', 'bank_shnu_vol': '은행_매수수량',
        'insu_seln_vol': '보험_매도수량', 'insu_shnu_vol': '보험_매수수량',
        'mrbn_seln_vol': '기타금융_매도수량', 'mrbn_shnu_vol': '기타금융_매수수량',
        'fund_seln_vol': '연기금_매도수량', 'fund_shnu_vol': '연기금_매수수량',
        'etc_seln_vol': '기타법인_매도수량', 'etc_shnu_vol': '기타법인_매수수량',
        'etc_corp_seln_vol': '기타법인_매도수량_2', 'etc_corp_shnu_vol': '기타법인_매수수량_2',
        'etc_orgt_seln_vol': '기타단체_매도수량', 'etc_orgt_shnu_vol': '기타단체_매수수량',

        # [세부 매도/매수 금액]
        'prsn_seln_tr_pbmn': '개인_매도금액', 'prsn_shnu_tr_pbmn': '개인_매수금액',
        'frgn_seln_tr_pbmn': '외국인_매도금액', 'frgn_shnu_tr_pbmn': '외국인_매수금액',
        'orgn_seln_tr_pbmn': '기관계_매도금액', 'orgn_shnu_tr_pbmn': '기관계_매수금액',
    }

    @classmethod
    def rename(cls, df):
        return df.rename(columns=cls.RENAME_MAP)