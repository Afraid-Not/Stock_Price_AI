"""
자동매매 메인 실행 파일
- 예측 AI + 리스크 관리 + 자동매매 통합
"""
import os
import sys
import time
import glob
import schedule
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import argparse
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("D:/stock/_v10/autotrader.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 모듈 임포트
from m01_model import StockPredictionModel
from m02_train import TrainingPipeline
from r01_risk_manager import RiskManager, TradeSignal
from t01_trader import KISTrader, PaperTrader
from s03_preprocessing import StockPreprocessor


# 종목 코드 매핑
STOCK_NAMES = {
    '005930': '삼성전자', '000660': 'SK하이닉스', '006400': '삼성SDI',
    '035420': 'NAVER', '035720': '카카오', '018260': '삼성에스디에스',
    '034220': 'LG디스플레이', '066570': 'LG전자', '030200': 'KT',
    '017670': 'SK텔레콤', '030520': '한글과컴퓨터', '000990': 'DB하이텍',
    '011070': 'LG이노텍', '012510': '더존비즈온', '032500': '케이엠더블유',
    '032640': 'LG유플러스', '036570': '엔씨소프트', '053800': '안랩',
    '067160': '아프리카TV', '078340': '컴투스', '112040': '위메이드',
    '218410': 'RFHIC', '222800': '심텍', '251270': '넷마블',
    '259960': '크래프톤', '263750': '펄어비스', '293490': '카카오게임즈',
    '336370': '솔루스첨단소재', '353200': '대덕전자', '402340': 'SK스퀘어'
}


class AutoTrader:
    """자동매매 시스템"""
    
    def __init__(self, config: dict = None):
        """
        config 옵션:
        - mode: 'paper' (모의거래) / 'mock' (증권사 모의투자) / 'real' (실전투자)
        - target_stocks: 대상 종목 리스트
        - capital: 총 자본금
        - model_path: 학습된 모델 경로
        """
        default_config = {
            'mode': 'paper',  # 기본은 페이퍼 트레이딩
            'target_stocks': list(STOCK_NAMES.keys())[:10],  # 상위 10종목
            'capital': 10_000_000,
            'model_path': None,
            'check_interval': 60,  # 시세 체크 간격 (초)
        }
        
        self.config = {**default_config, **(config or {})}
        self.model: StockPredictionModel = None
        self.risk_manager: RiskManager = None
        self.trader = None
        self.is_running = False
        
        # 전처리기 (예측용)
        self.preprocessor = StockPreprocessor()
        
        self._initialize()
    
    def _initialize(self):
        """초기화"""
        logger.info("=" * 60)
        logger.info("🚀 자동매매 시스템 초기화")
        logger.info("=" * 60)
        
        # 1. 모델 로드
        self._load_model()
        
        # 2. 리스크 관리자 초기화
        self.risk_manager = RiskManager(config={
            'total_capital': self.config['capital']
        })
        
        # 저장된 상태 로드
        try:
            self.risk_manager.load_state()
        except:
            pass
        
        # 3. 트레이더 초기화
        mode = self.config['mode']
        if mode == 'paper':
            self.trader = PaperTrader(self.config['capital'])
            logger.info("📝 페이퍼 트레이딩 모드")
        elif mode == 'mock':
            self.trader = KISTrader(is_mock=True)
            logger.info("🎮 모의투자 모드")
        else:
            self.trader = KISTrader(is_mock=False)
            logger.info("💰 실전투자 모드")
        
        logger.info(f"💼 대상 종목: {len(self.config['target_stocks'])}개")
        logger.info(f"💵 자본금: {self.config['capital']:,}원")
    
    def _load_model(self):
        """모델 로드"""
        model_path = self.config.get('model_path')
        
        if model_path and os.path.exists(model_path):
            self.model = StockPredictionModel()
            self.model.load(model_path)
        else:
            # 가장 최근 모델 찾기
            model_dir = "D:/stock/_v10/models"
            if os.path.exists(model_dir):
                model_files = glob.glob(os.path.join(model_dir, "ensemble_model_*.pkl"))
                if model_files:
                    latest_model = max(model_files, key=os.path.getctime)
                    self.model = StockPredictionModel()
                    self.model.load(latest_model)
                    return
            
            logger.warning("⚠️ 학습된 모델이 없습니다. 먼저 학습을 진행하세요.")
            logger.info("   python m02_train.py --all")
    
    def _get_prediction_features(self, stock_code: str) -> pd.DataFrame:
        """예측용 피처 생성 (최신 데이터 기반)"""
        # 전처리된 데이터 로드
        preprocessed_dir = "D:/stock/_v10/_preprocessed"
        final_file = os.path.join(preprocessed_dir, f"{stock_code}_final.csv")
        
        if not os.path.exists(final_file):
            # 원본 데이터에서 전처리
            data_dir = "D:/stock/_v10/_data/stock"
            pattern = os.path.join(data_dir, f"{stock_code}_*.csv")
            files = glob.glob(pattern)
            
            if not files:
                return None
            
            from s02_rename import rename_file
            renamed_file = final_file.replace("_final.csv", "_renamed.csv")
            rename_file(files[0], renamed_file)
            
            self.preprocessor.stock_code = stock_code
            self.preprocessor.run_pipeline(renamed_file, final_file, is_train=False)
        
        df = pd.read_csv(final_file)
        
        # 마지막 행의 피처 반환 (최신 데이터)
        exclude_cols = ['날짜', 'target', 'next_rtn', 'stock_code']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return df[feature_cols].iloc[[-1]]  # 마지막 행
    
    def _get_stock_volatility(self, stock_code: str) -> float:
        """종목 변동성 계산 (최근 20일 기준)"""
        preprocessed_dir = "D:/stock/_v10/_preprocessed"
        final_file = os.path.join(preprocessed_dir, f"{stock_code}_final.csv")
        
        if os.path.exists(final_file):
            df = pd.read_csv(final_file)
            if 'volatility' in df.columns:
                return df['volatility'].iloc[-20:].mean()
        
        return 0.02  # 기본 변동성
    
    def analyze_stock(self, stock_code: str) -> Dict:
        """종목 분석 및 시그널 생성"""
        stock_name = STOCK_NAMES.get(stock_code, stock_code)
        
        # 1. 현재가 조회
        if isinstance(self.trader, PaperTrader):
            price_info = self.trader.get_current_price(stock_code)
        else:
            price_info = self.trader.get_current_price(stock_code)
        
        if not price_info:
            return {'stock_code': stock_code, 'signal': None, 'error': '현재가 조회 실패'}
        
        current_price = price_info['price']
        
        # 2. 예측 수행
        if self.model is None:
            return {'stock_code': stock_code, 'signal': None, 'error': '모델 없음'}
        
        features = self._get_prediction_features(stock_code)
        if features is None:
            return {'stock_code': stock_code, 'signal': None, 'error': '피처 생성 실패'}
        
        try:
            prediction_proba = self.model.predict_proba(features.fillna(0))[0]
        except Exception as e:
            return {'stock_code': stock_code, 'signal': None, 'error': f'예측 실패: {e}'}
        
        # 3. 변동성 계산
        volatility = self._get_stock_volatility(stock_code)
        
        # 4. 시그널 생성
        signal = self.risk_manager.generate_signal(
            stock_code=stock_code,
            stock_name=stock_name,
            prediction_proba=prediction_proba,
            current_price=current_price,
            volatility=volatility
        )
        
        return {
            'stock_code': stock_code,
            'stock_name': stock_name,
            'current_price': current_price,
            'prediction_proba': prediction_proba,
            'signal': signal
        }
    
    def execute_trades(self):
        """전체 종목 분석 및 매매 실행"""
        logger.info("\n" + "=" * 60)
        logger.info(f"📊 매매 분석 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        results = []
        
        for stock_code in self.config['target_stocks']:
            try:
                analysis = self.analyze_stock(stock_code)
                
                if analysis.get('error'):
                    logger.warning(f"⚠️ {stock_code}: {analysis['error']}")
                    continue
                
                signal = analysis['signal']
                
                if signal and signal.action != 'HOLD':
                    logger.info(f"\n🎯 {analysis['stock_name']} ({stock_code})")
                    logger.info(f"   현재가: {analysis['current_price']:,}원")
                    logger.info(f"   상승 확률: {analysis['prediction_proba']:.1%}")
                    logger.info(f"   시그널: {signal.action} ({signal.reason})")
                    
                    if signal.action == 'BUY':
                        logger.info(f"   수량: {signal.target_quantity}주")
                        logger.info(f"   손절가: {signal.stop_loss:,.0f}원")
                        logger.info(f"   익절가: {signal.take_profit:,.0f}원")
                    
                    # 매매 실행
                    if isinstance(self.trader, PaperTrader):
                        result = self.trader.execute_signal(signal)
                    else:
                        result = self.trader.execute_signal(signal, use_market_price=True)
                    
                    if result and result.get('success'):
                        if signal.action == 'BUY':
                            self.risk_manager.add_position(signal)
                        elif signal.action == 'SELL':
                            self.risk_manager.close_position(signal, analysis['current_price'])
                    
                    results.append({
                        'stock_code': stock_code,
                        'action': signal.action,
                        'result': result
                    })
                
                time.sleep(0.5)  # API 호출 제한
                
            except Exception as e:
                logger.error(f"❌ {stock_code} 처리 중 오류: {e}")
        
        # 포트폴리오 요약
        summary = self.risk_manager.get_portfolio_summary()
        logger.info("\n" + "-" * 40)
        logger.info("📈 포트폴리오 현황")
        logger.info(f"   보유 종목: {summary['position_count']}개")
        logger.info(f"   투자 금액: {summary['total_position_value']:,.0f}원")
        logger.info(f"   미실현 손익: {summary['total_unrealized_pnl']:+,.0f}원")
        logger.info(f"   일일 실현 손익: {summary['daily_pnl']:+,.0f}원")
        
        # 상태 저장
        self.risk_manager.save_state()
        
        return results
    
    def run_once(self):
        """1회 실행"""
        return self.execute_trades()
    
    def run_loop(self, interval_seconds: int = None):
        """반복 실행 (장중 모니터링)"""
        interval = interval_seconds or self.config['check_interval']
        
        logger.info(f"🔄 자동매매 루프 시작 (간격: {interval}초)")
        self.is_running = True
        
        while self.is_running:
            try:
                # 장 시간 체크 (09:00 ~ 15:20)
                now = datetime.now()
                market_open = now.replace(hour=9, minute=0, second=0)
                market_close = now.replace(hour=15, minute=20, second=0)
                
                if market_open <= now <= market_close:
                    self.execute_trades()
                else:
                    logger.info(f"⏰ 장외 시간입니다. 대기 중... ({now.strftime('%H:%M')})")
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 사용자에 의해 중지됨")
                self.is_running = False
            except Exception as e:
                logger.error(f"❌ 루프 오류: {e}")
                time.sleep(60)
    
    def stop(self):
        """실행 중지"""
        self.is_running = False
        self.risk_manager.save_state()
        logger.info("🛑 자동매매 중지")


def main():
    parser = argparse.ArgumentParser(description="주가 예측 AI 자동매매 시스템")
    
    parser.add_argument("--mode", choices=['paper', 'mock', 'real'], default='paper',
                       help="매매 모드 (paper: 모의, mock: 증권사모의, real: 실전)")
    parser.add_argument("--capital", type=int, default=10_000_000,
                       help="투자 자본금 (기본: 1000만원)")
    parser.add_argument("--stocks", nargs="+", default=None,
                       help="대상 종목 코드 (예: 005930 000660)")
    parser.add_argument("--once", action="store_true",
                       help="1회만 실행")
    parser.add_argument("--interval", type=int, default=300,
                       help="실행 간격 (초, 기본: 300)")
    parser.add_argument("--train", action="store_true",
                       help="모델 학습 먼저 실행")
    
    args = parser.parse_args()
    
    # 모델 학습
    if args.train:
        logger.info("📚 모델 학습 시작...")
        pipeline = TrainingPipeline()
        pipeline.run_full_pipeline(preprocess=True)
        logger.info("✅ 모델 학습 완료\n")
    
    # 자동매매 설정
    config = {
        'mode': args.mode,
        'capital': args.capital,
        'check_interval': args.interval
    }
    
    if args.stocks:
        config['target_stocks'] = [s.zfill(6) for s in args.stocks]
    
    # 실행
    auto_trader = AutoTrader(config)
    
    if args.once:
        auto_trader.run_once()
    else:
        auto_trader.run_loop()


if __name__ == "__main__":
    main()

