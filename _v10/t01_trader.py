"""
자동매매 트레이더 - 한국투자증권 API 연동
"""
import requests
import json
import time
import os
from datetime import datetime
from typing import Optional, Dict, List
from dotenv import load_dotenv
from s00_get_token import get_access_token
from r01_risk_manager import TradeSignal

load_dotenv()


class KISTrader:
    """한국투자증권 API 기반 트레이더"""
    
    def __init__(self, is_mock: bool = True):
        """
        is_mock: True면 모의투자, False면 실전투자
        """
        self.is_mock = is_mock
        
        # API 설정
        if is_mock:
            self.app_key = os.getenv("MOCK_APP_KEY")
            self.app_secret = os.getenv("MOCK_APP_SECRET")
            self.base_url = "https://openapivts.koreainvestment.com:29443"
            self.account_no = os.getenv("MOCK_ACCOUNT_NO", "")
        else:
            self.app_key = os.getenv("REAL_APP_KEY")
            self.app_secret = os.getenv("REAL_APP_SECRET")
            self.base_url = "https://openapi.koreainvestment.com:9443"
            self.account_no = os.getenv("REAL_ACCOUNT_NO", "")
        
        # 계좌번호 파싱
        if self.account_no:
            parts = self.account_no.split("-")
            self.cano = parts[0] if len(parts) > 0 else ""
            self.acnt_prdt_cd = parts[1] if len(parts) > 1 else "01"
        else:
            self.cano = ""
            self.acnt_prdt_cd = "01"
        
        self.token = None
        self._refresh_token()
    
    def _refresh_token(self):
        """토큰 갱신"""
        self.token = get_access_token()
        if not self.token:
            raise ValueError("토큰 발급 실패! .env 파일을 확인하세요.")
    
    def _get_headers(self, tr_id: str) -> dict:
        """API 요청 헤더"""
        return {
            "Content-Type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self.token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id
        }
    
    def get_current_price(self, stock_code: str) -> Optional[Dict]:
        """현재가 조회"""
        path = "/uapi/domestic-stock/v1/quotations/inquire-price"
        url = f"{self.base_url}{path}"
        
        tr_id = "FHKST01010100"
        
        headers = self._get_headers(tr_id)
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": stock_code
        }
        
        try:
            res = requests.get(url, headers=headers, params=params, timeout=10)
            data = res.json()
            
            if data.get('rt_cd') == '0':
                output = data.get('output', {})
                return {
                    'price': int(output.get('stck_prpr', 0)),
                    'change': int(output.get('prdy_vrss', 0)),
                    'change_rate': float(output.get('prdy_ctrt', 0)),
                    'volume': int(output.get('acml_vol', 0)),
                    'high': int(output.get('stck_hgpr', 0)),
                    'low': int(output.get('stck_lwpr', 0)),
                    'open': int(output.get('stck_oprc', 0)),
                }
            else:
                print(f"⚠️ 현재가 조회 실패: {data.get('msg1', '')}")
                return None
                
        except Exception as e:
            print(f"❌ 현재가 조회 오류: {e}")
            return None
    
    def get_balance(self) -> Optional[Dict]:
        """주식 잔고 조회"""
        path = "/uapi/domestic-stock/v1/trading/inquire-balance"
        url = f"{self.base_url}{path}"
        
        tr_id = "VTTC8434R" if self.is_mock else "TTTC8434R"
        
        headers = self._get_headers(tr_id)
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        }
        
        try:
            res = requests.get(url, headers=headers, params=params, timeout=10)
            data = res.json()
            
            if data.get('rt_cd') == '0':
                output1 = data.get('output1', [])
                output2 = data.get('output2', [{}])[0]
                
                holdings = []
                for item in output1:
                    if int(item.get('hldg_qty', 0)) > 0:
                        holdings.append({
                            'stock_code': item.get('pdno', ''),
                            'stock_name': item.get('prdt_name', ''),
                            'quantity': int(item.get('hldg_qty', 0)),
                            'avg_price': float(item.get('pchs_avg_pric', 0)),
                            'current_price': int(item.get('prpr', 0)),
                            'eval_amount': int(item.get('evlu_amt', 0)),
                            'profit_loss': int(item.get('evlu_pfls_amt', 0)),
                            'profit_rate': float(item.get('evlu_pfls_rt', 0))
                        })
                
                return {
                    'holdings': holdings,
                    'total_eval': int(output2.get('tot_evlu_amt', 0)),
                    'total_profit': int(output2.get('evlu_pfls_smtl_amt', 0)),
                    'cash': int(output2.get('dnca_tot_amt', 0))
                }
            else:
                print(f"⚠️ 잔고 조회 실패: {data.get('msg1', '')}")
                return None
                
        except Exception as e:
            print(f"❌ 잔고 조회 오류: {e}")
            return None
    
    def get_buyable_amount(self, stock_code: str, price: int) -> int:
        """매수 가능 금액 조회"""
        path = "/uapi/domestic-stock/v1/trading/inquire-psbl-order"
        url = f"{self.base_url}{path}"
        
        tr_id = "VTTC8908R" if self.is_mock else "TTTC8908R"
        
        headers = self._get_headers(tr_id)
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": stock_code,
            "ORD_UNPR": str(price),
            "ORD_DVSN": "00",
            "CMA_EVLU_AMT_ICLD_YN": "Y",
            "OVRS_ICLD_YN": "N"
        }
        
        try:
            res = requests.get(url, headers=headers, params=params, timeout=10)
            data = res.json()
            
            if data.get('rt_cd') == '0':
                output = data.get('output', {})
                return int(output.get('nrcvb_buy_amt', 0))
            return 0
            
        except Exception as e:
            print(f"❌ 매수가능금액 조회 오류: {e}")
            return 0
    
    def order_stock(self, stock_code: str, quantity: int, price: int = 0,
                   order_type: str = "BUY", price_type: str = "00") -> Optional[Dict]:
        """
        주식 주문
        
        order_type: BUY(매수) / SELL(매도)
        price_type: 
            00 - 지정가
            01 - 시장가  
            03 - 최유리지정가
            05 - 최우선지정가
        """
        path = "/uapi/domestic-stock/v1/trading/order-cash"
        url = f"{self.base_url}{path}"
        
        # TR_ID 설정
        if self.is_mock:
            tr_id = "VTTC0802U" if order_type == "BUY" else "VTTC0801U"
        else:
            tr_id = "TTTC0802U" if order_type == "BUY" else "TTTC0801U"
        
        headers = self._get_headers(tr_id)
        
        body = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "PDNO": stock_code,
            "ORD_DVSN": price_type,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price) if price_type == "00" else "0"
        }
        
        try:
            res = requests.post(url, headers=headers, data=json.dumps(body), timeout=10)
            data = res.json()
            
            if data.get('rt_cd') == '0':
                output = data.get('output', {})
                order_no = output.get('ODNO', '')
                
                print(f"✅ 주문 성공: {order_type} {stock_code} {quantity}주 @ {price}원 (주문번호: {order_no})")
                
                return {
                    'success': True,
                    'order_no': order_no,
                    'stock_code': stock_code,
                    'quantity': quantity,
                    'price': price,
                    'order_type': order_type
                }
            else:
                print(f"⚠️ 주문 실패: {data.get('msg1', '')}")
                return {
                    'success': False,
                    'message': data.get('msg1', '알 수 없는 오류')
                }
                
        except Exception as e:
            print(f"❌ 주문 오류: {e}")
            return {'success': False, 'message': str(e)}
    
    def buy(self, stock_code: str, quantity: int, price: int = 0, 
            use_market_price: bool = True) -> Optional[Dict]:
        """매수 주문"""
        price_type = "01" if use_market_price else "00"
        return self.order_stock(stock_code, quantity, price, "BUY", price_type)
    
    def sell(self, stock_code: str, quantity: int, price: int = 0,
             use_market_price: bool = True) -> Optional[Dict]:
        """매도 주문"""
        price_type = "01" if use_market_price else "00"
        return self.order_stock(stock_code, quantity, price, "SELL", price_type)
    
    def execute_signal(self, signal: TradeSignal, use_market_price: bool = True) -> Optional[Dict]:
        """매매 시그널 실행"""
        if signal.action == 'HOLD':
            return None
        
        if signal.action == 'BUY':
            return self.buy(
                signal.stock_code, 
                signal.target_quantity, 
                int(signal.target_price),
                use_market_price
            )
        elif signal.action == 'SELL':
            return self.sell(
                signal.stock_code,
                signal.target_quantity,
                int(signal.target_price),
                use_market_price
            )
        
        return None


class PaperTrader:
    """모의 트레이더 (실제 주문 없이 시뮬레이션)"""
    
    def __init__(self, initial_capital: float = 10_000_000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings: Dict[str, Dict] = {}
        self.orders: List[Dict] = []
    
    def get_current_price(self, stock_code: str) -> Optional[Dict]:
        """실제 API로 현재가 조회"""
        trader = KISTrader(is_mock=True)
        return trader.get_current_price(stock_code)
    
    def buy(self, stock_code: str, stock_name: str, quantity: int, price: float) -> Dict:
        """모의 매수"""
        total_cost = quantity * price
        
        if total_cost > self.cash:
            return {'success': False, 'message': '잔고 부족'}
        
        self.cash -= total_cost
        
        if stock_code in self.holdings:
            existing = self.holdings[stock_code]
            new_qty = existing['quantity'] + quantity
            new_avg = (existing['quantity'] * existing['avg_price'] + total_cost) / new_qty
            existing['quantity'] = new_qty
            existing['avg_price'] = new_avg
        else:
            self.holdings[stock_code] = {
                'stock_name': stock_name,
                'quantity': quantity,
                'avg_price': price
            }
        
        order = {
            'time': datetime.now().isoformat(),
            'action': 'BUY',
            'stock_code': stock_code,
            'stock_name': stock_name,
            'quantity': quantity,
            'price': price,
            'total': total_cost
        }
        self.orders.append(order)
        
        print(f"📈 [모의매수] {stock_name} {quantity}주 @ {price:,.0f}원")
        return {'success': True, 'order': order}
    
    def sell(self, stock_code: str, quantity: int, price: float) -> Dict:
        """모의 매도"""
        if stock_code not in self.holdings:
            return {'success': False, 'message': '보유 종목 없음'}
        
        holding = self.holdings[stock_code]
        if holding['quantity'] < quantity:
            return {'success': False, 'message': '보유 수량 부족'}
        
        total_revenue = quantity * price
        pnl = (price - holding['avg_price']) * quantity
        
        self.cash += total_revenue
        holding['quantity'] -= quantity
        
        if holding['quantity'] == 0:
            del self.holdings[stock_code]
        
        order = {
            'time': datetime.now().isoformat(),
            'action': 'SELL',
            'stock_code': stock_code,
            'stock_name': holding.get('stock_name', stock_code),
            'quantity': quantity,
            'price': price,
            'total': total_revenue,
            'pnl': pnl
        }
        self.orders.append(order)
        
        print(f"📉 [모의매도] {stock_code} {quantity}주 @ {price:,.0f}원 (손익: {pnl:+,.0f}원)")
        return {'success': True, 'order': order, 'pnl': pnl}
    
    def execute_signal(self, signal: TradeSignal) -> Optional[Dict]:
        """시그널 실행"""
        if signal.action == 'HOLD':
            return None
        
        if signal.action == 'BUY':
            return self.buy(
                signal.stock_code,
                signal.stock_name,
                signal.target_quantity,
                signal.target_price
            )
        elif signal.action == 'SELL':
            return self.sell(
                signal.stock_code,
                signal.target_quantity,
                signal.target_price
            )
        
        return None
    
    def get_portfolio_value(self, price_dict: Dict[str, float] = None) -> float:
        """포트폴리오 총 가치"""
        holdings_value = 0
        for code, holding in self.holdings.items():
            price = price_dict.get(code, holding['avg_price']) if price_dict else holding['avg_price']
            holdings_value += holding['quantity'] * price
        
        return self.cash + holdings_value
    
    def get_summary(self) -> Dict:
        """요약 정보"""
        return {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'holdings': self.holdings,
            'total_trades': len(self.orders)
        }


if __name__ == "__main__":
    # 테스트
    print("=" * 60)
    print("🧪 트레이더 테스트")
    print("=" * 60)
    
    # 모의투자 트레이더 테스트
    trader = KISTrader(is_mock=True)
    
    # 현재가 조회
    price_info = trader.get_current_price("005930")
    if price_info:
        print(f"\n삼성전자 현재가: {price_info['price']:,}원")
        print(f"등락률: {price_info['change_rate']:+.2f}%")
    
    # 잔고 조회
    balance = trader.get_balance()
    if balance:
        print(f"\n예수금: {balance['cash']:,}원")
        print(f"총평가: {balance['total_eval']:,}원")

