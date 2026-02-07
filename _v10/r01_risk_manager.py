"""
리스크 관리 모듈
- 포지션 사이징 (켈리 공식)
- 손절/익절 관리
- 최대 손실 제한
- 포트폴리오 분산
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
import os


@dataclass
class Position:
    """포지션 정보"""
    stock_code: str
    stock_name: str
    quantity: int
    avg_price: float
    current_price: float
    entry_date: datetime
    stop_loss: float = 0.0  # 손절가
    take_profit: float = 0.0  # 익절가
    
    @property
    def market_value(self) -> float:
        """시장 가치"""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """매수 금액"""
        return self.quantity * self.avg_price
    
    @property
    def unrealized_pnl(self) -> float:
        """미실현 손익"""
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """미실현 손익률"""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100


@dataclass
class TradeSignal:
    """매매 시그널"""
    stock_code: str
    stock_name: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0~1 예측 확률
    target_quantity: int = 0
    target_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    reason: str = ""


class RiskManager:
    """리스크 관리자"""
    
    def __init__(self, config: dict = None):
        """
        config 옵션:
        - total_capital: 총 자본금
        - max_single_position_pct: 단일 종목 최대 비중 (기본 20%)
        - max_total_position_pct: 총 투자 비중 (기본 80%)
        - max_daily_loss_pct: 일일 최대 손실률 (기본 3%)
        - default_stop_loss_pct: 기본 손절률 (기본 5%)
        - default_take_profit_pct: 기본 익절률 (기본 10%)
        - min_confidence: 최소 신뢰도 (기본 0.55)
        """
        default_config = {
            'total_capital': 10_000_000,
            'max_single_position_pct': 0.20,
            'max_total_position_pct': 0.80,
            'max_daily_loss_pct': 0.03,
            'default_stop_loss_pct': 0.05,
            'default_take_profit_pct': 0.10,
            'min_confidence': 0.55,
            'kelly_fraction': 0.5,  # 켈리 비율의 절반 (보수적)
        }
        
        self.config = {**default_config, **(config or {})}
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.daily_trades: List[dict] = []
        self.trade_history: List[dict] = []
        
    def calculate_position_size(self, confidence: float, current_price: float, 
                                 volatility: float = 0.02) -> int:
        """
        켈리 공식 기반 포지션 사이즈 계산
        
        Kelly Fraction = (bp - q) / b
        - b: 수익 비율 (보통 1:1 가정)
        - p: 승률 (confidence)
        - q: 패률 (1 - confidence)
        """
        if confidence < self.config['min_confidence']:
            return 0
        
        # 켈리 공식
        p = confidence
        q = 1 - confidence
        b = 1  # 손익비 1:1 가정
        
        kelly = (b * p - q) / b
        kelly = max(0, kelly)  # 음수 방지
        
        # 보수적 켈리 (절반만 사용)
        kelly *= self.config['kelly_fraction']
        
        # 변동성 조정 (변동성 높으면 더 적게 투자)
        vol_adjustment = 0.02 / max(volatility, 0.01)
        kelly *= min(vol_adjustment, 1.5)
        
        # 최대 단일 포지션 비중 제한
        kelly = min(kelly, self.config['max_single_position_pct'])
        
        # 투자 금액 계산
        available_capital = self._get_available_capital()
        invest_amount = available_capital * kelly
        
        # 수량 계산
        quantity = int(invest_amount / current_price)
        
        return max(0, quantity)
    
    def _get_available_capital(self) -> float:
        """투자 가능 자본 계산"""
        total_capital = self.config['total_capital']
        max_invest = total_capital * self.config['max_total_position_pct']
        
        # 현재 보유 포지션 가치
        current_position_value = sum(pos.market_value for pos in self.positions.values())
        
        available = max_invest - current_position_value
        return max(0, available)
    
    def calculate_stop_loss(self, entry_price: float, atr: float = None) -> float:
        """손절가 계산 (ATR 기반 동적 손절)"""
        if atr:
            # ATR의 2배를 손절폭으로 사용
            stop_loss = entry_price - (atr * 2)
        else:
            # 기본 손절률 사용
            stop_loss = entry_price * (1 - self.config['default_stop_loss_pct'])
        
        return round(stop_loss, 0)
    
    def calculate_take_profit(self, entry_price: float, atr: float = None) -> float:
        """익절가 계산"""
        if atr:
            # ATR의 3배를 익절폭으로 사용 (손익비 1.5:1)
            take_profit = entry_price + (atr * 3)
        else:
            # 기본 익절률 사용
            take_profit = entry_price * (1 + self.config['default_take_profit_pct'])
        
        return round(take_profit, 0)
    
    def generate_signal(self, stock_code: str, stock_name: str, 
                        prediction_proba: float, current_price: float,
                        volatility: float = 0.02, atr: float = None) -> TradeSignal:
        """매매 시그널 생성"""
        
        # 기존 포지션 확인
        has_position = stock_code in self.positions
        
        # 일일 손실 한도 체크
        if self._is_daily_loss_exceeded():
            return TradeSignal(
                stock_code=stock_code,
                stock_name=stock_name,
                action='HOLD',
                confidence=prediction_proba,
                reason="일일 최대 손실 한도 도달"
            )
        
        # 매수 시그널
        if prediction_proba >= self.config['min_confidence'] and not has_position:
            quantity = self.calculate_position_size(
                prediction_proba, current_price, volatility
            )
            
            if quantity > 0:
                stop_loss = self.calculate_stop_loss(current_price, atr)
                take_profit = self.calculate_take_profit(current_price, atr)
                
                return TradeSignal(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    action='BUY',
                    confidence=prediction_proba,
                    target_quantity=quantity,
                    target_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"상승 확률 {prediction_proba:.1%}"
                )
        
        # 매도 시그널 (보유 중인 경우)
        if has_position:
            position = self.positions[stock_code]
            position.current_price = current_price
            
            # 손절 체크
            if current_price <= position.stop_loss:
                return TradeSignal(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    action='SELL',
                    confidence=prediction_proba,
                    target_quantity=position.quantity,
                    target_price=current_price,
                    reason=f"손절 ({position.unrealized_pnl_pct:.1f}%)"
                )
            
            # 익절 체크
            if current_price >= position.take_profit:
                return TradeSignal(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    action='SELL',
                    confidence=prediction_proba,
                    target_quantity=position.quantity,
                    target_price=current_price,
                    reason=f"익절 ({position.unrealized_pnl_pct:.1f}%)"
                )
            
            # 하락 예측 시 매도
            if prediction_proba < 0.45:
                return TradeSignal(
                    stock_code=stock_code,
                    stock_name=stock_name,
                    action='SELL',
                    confidence=prediction_proba,
                    target_quantity=position.quantity,
                    target_price=current_price,
                    reason=f"하락 예측 ({prediction_proba:.1%})"
                )
        
        # 관망
        return TradeSignal(
            stock_code=stock_code,
            stock_name=stock_name,
            action='HOLD',
            confidence=prediction_proba,
            reason="조건 미충족"
        )
    
    def _is_daily_loss_exceeded(self) -> bool:
        """일일 손실 한도 초과 여부"""
        max_loss = self.config['total_capital'] * self.config['max_daily_loss_pct']
        return self.daily_pnl < -max_loss
    
    def add_position(self, signal: TradeSignal) -> bool:
        """포지션 추가"""
        if signal.action != 'BUY':
            return False
        
        position = Position(
            stock_code=signal.stock_code,
            stock_name=signal.stock_name,
            quantity=signal.target_quantity,
            avg_price=signal.target_price,
            current_price=signal.target_price,
            entry_date=datetime.now(),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        self.positions[signal.stock_code] = position
        self._record_trade('BUY', signal)
        
        return True
    
    def close_position(self, signal: TradeSignal, execution_price: float) -> float:
        """포지션 청산"""
        if signal.stock_code not in self.positions:
            return 0.0
        
        position = self.positions[signal.stock_code]
        position.current_price = execution_price
        
        realized_pnl = position.unrealized_pnl
        self.daily_pnl += realized_pnl
        
        self._record_trade('SELL', signal, realized_pnl)
        
        del self.positions[signal.stock_code]
        
        return realized_pnl
    
    def _record_trade(self, action: str, signal: TradeSignal, pnl: float = 0.0):
        """거래 기록"""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'stock_code': signal.stock_code,
            'stock_name': signal.stock_name,
            'action': action,
            'quantity': signal.target_quantity,
            'price': signal.target_price,
            'confidence': signal.confidence,
            'reason': signal.reason,
            'pnl': pnl
        }
        
        self.daily_trades.append(trade)
        self.trade_history.append(trade)
    
    def reset_daily(self):
        """일일 데이터 초기화 (매일 장 시작 전)"""
        self.daily_pnl = 0.0
        self.daily_trades = []
    
    def get_portfolio_summary(self) -> dict:
        """포트폴리오 요약"""
        total_position_value = sum(pos.market_value for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        return {
            'total_capital': self.config['total_capital'],
            'available_capital': self._get_available_capital(),
            'total_position_value': total_position_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'position_count': len(self.positions),
            'daily_pnl': self.daily_pnl,
            'positions': {
                code: {
                    'name': pos.stock_name,
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct
                }
                for code, pos in self.positions.items()
            }
        }
    
    def save_state(self, filepath: str = "D:/stock/_v10/risk_state.json"):
        """상태 저장"""
        state = {
            'config': self.config,
            'positions': {
                code: {
                    'stock_code': pos.stock_code,
                    'stock_name': pos.stock_name,
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'current_price': pos.current_price,
                    'entry_date': pos.entry_date.isoformat(),
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit
                }
                for code, pos in self.positions.items()
            },
            'trade_history': self.trade_history[-100:],  # 최근 100건만
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        
        print(f"💾 리스크 상태 저장: {filepath}")
    
    def load_state(self, filepath: str = "D:/stock/_v10/risk_state.json"):
        """상태 로드"""
        if not os.path.exists(filepath):
            print(f"⚠️ 상태 파일 없음: {filepath}")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.config = state.get('config', self.config)
        self.trade_history = state.get('trade_history', [])
        
        # 포지션 복원
        for code, pos_data in state.get('positions', {}).items():
            self.positions[code] = Position(
                stock_code=pos_data['stock_code'],
                stock_name=pos_data['stock_name'],
                quantity=pos_data['quantity'],
                avg_price=pos_data['avg_price'],
                current_price=pos_data['current_price'],
                entry_date=datetime.fromisoformat(pos_data['entry_date']),
                stop_loss=pos_data['stop_loss'],
                take_profit=pos_data['take_profit']
            )
        
        print(f"📂 리스크 상태 로드: {filepath}")


if __name__ == "__main__":
    # 테스트
    rm = RiskManager(config={'total_capital': 10_000_000})
    
    # 시그널 생성 테스트
    signal = rm.generate_signal(
        stock_code="005930",
        stock_name="삼성전자",
        prediction_proba=0.72,
        current_price=58000,
        volatility=0.025
    )
    
    print(f"시그널: {signal.action}")
    print(f"수량: {signal.target_quantity}")
    print(f"손절가: {signal.stop_loss}")
    print(f"익절가: {signal.take_profit}")
    print(f"사유: {signal.reason}")

