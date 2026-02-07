# 🤖 주가 예측 AI + 자동매매 시스템

한국투자증권 API를 활용한 주가 예측 AI와 리스크 관리 기반 자동매매 프로그램

## 📁 파일 구조

```
D:\stock\_v10\
├── 📊 데이터 수집/전처리
│   ├── s00_get_token.py      # API 토큰 발급
│   ├── s01_kis_data_get.py   # 주식 데이터 수집
│   ├── s02_rename.py         # 컬럼명 한글화
│   ├── s03_preprocessing.py  # 피처 엔지니어링
│   └── stock_utils.py        # 유틸리티
│
├── 🧠 예측 AI 모듈
│   ├── m01_model.py          # 앙상블 모델 (XGBoost + LightGBM + CatBoost)
│   └── m02_train.py          # 학습 파이프라인
│
├── 🛡️ 리스크 관리 모듈
│   └── r01_risk_manager.py   # 포지션 사이징, 손절/익절, 포트폴리오 관리
│
├── 💰 자동매매 모듈
│   └── t01_trader.py         # 한국투자증권 API 주문 실행
│
└── 🚀 메인 실행
    └── main_autotrader.py    # 통합 자동매매 시스템
```

## ⚙️ 설치 및 설정

### 1. 의존성 설치
```bash
cd D:\stock\_v10
conda activate <your_env>
pip install -r requirements.txt
```

### 2. 환경변수 설정 (.env)
```env
# 실전투자
REAL_APP_KEY=your_real_app_key
REAL_APP_SECRET=your_real_app_secret
REAL_ACCOUNT_NO=12345678-01

# 모의투자
MOCK_APP_KEY=your_mock_app_key
MOCK_APP_SECRET=your_mock_app_secret
MOCK_ACCOUNT_NO=12345678-01
```

## 🎯 사용법

### Step 1: 모델 학습
```bash
# 전체 종목 학습
python m02_train.py --all

# 단일 종목 학습
python m02_train.py --stock 005930
```

### Step 2: 자동매매 실행
```bash
# 페이퍼 트레이딩 (실제 주문 없음, 테스트용)
python main_autotrader.py --mode paper --once

# 모의투자 (증권사 모의투자 계좌)
python main_autotrader.py --mode mock --once

# 실전투자 (실제 계좌)
python main_autotrader.py --mode real --once
```

### 옵션 설명
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--mode` | 매매 모드 (paper/mock/real) | paper |
| `--capital` | 투자 자본금 | 10,000,000원 |
| `--stocks` | 대상 종목 (공백 구분) | 상위 10종목 |
| `--once` | 1회만 실행 | False (반복) |
| `--interval` | 실행 간격 (초) | 300초 |
| `--train` | 학습 먼저 실행 | False |

### 예시
```bash
# 삼성전자, SK하이닉스만 5분 간격으로 모의투자
python main_autotrader.py --mode mock --stocks 005930 000660 --interval 300

# 2000만원으로 실전투자 (1회 실행)
python main_autotrader.py --mode real --capital 20000000 --once

# 학습 후 바로 자동매매 시작
python main_autotrader.py --train --mode paper
```

## 📈 핵심 기능

### 1. 예측 AI (앙상블 모델)
- **XGBoost + LightGBM + CatBoost** 3종 앙상블
- 성능 기반 동적 가중치 조정
- 다음 날 상승/하락 확률 예측

### 2. 리스크 관리
- **켈리 공식** 기반 포지션 사이징 (보수적 절반 적용)
- **ATR 기반** 동적 손절/익절가 설정
- 일일 최대 손실 한도 (기본 3%)
- 단일 종목 최대 비중 제한 (기본 20%)

### 3. 매매 규칙
- **매수 조건**: 상승 확률 55% 이상 + 포지션 미보유
- **매도 조건**: 손절가 도달 / 익절가 도달 / 하락 예측 (45% 미만)

## ⚠️ 주의사항

1. **실전투자 전 반드시 페이퍼/모의투자로 충분히 테스트하세요**
2. 과거 성과가 미래 수익을 보장하지 않습니다
3. 투자 손실에 대한 책임은 본인에게 있습니다
4. API 호출 제한에 주의하세요 (초당 요청 수 제한)

## 📊 로그 확인
```bash
# 실시간 로그 확인
tail -f D:\stock\_v10\autotrader.log
```

## 🔧 커스터마이징

### 리스크 설정 변경
```python
# r01_risk_manager.py
config = {
    'max_single_position_pct': 0.15,  # 단일 종목 15%로 변경
    'default_stop_loss_pct': 0.03,    # 손절 3%로 변경
    'min_confidence': 0.60,            # 최소 신뢰도 60%로 변경
}
```

### 모델 하이퍼파라미터 변경
```python
# m01_model.py의 _get_xgb_model() 등에서 수정
```

---
Made with ❤️ for 재현님

