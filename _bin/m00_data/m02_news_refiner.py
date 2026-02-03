import pandas as pd
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import warnings
from datetime import datetime, timedelta
import numpy as np
import math

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# FinBERT를 사용한 뉴스 감성 분석 (테마 필터링 기능 추가)
# ---------------------------------------------------------

def load_sentiment_model():
    """
    한국어 금융 감성 분석 모델 로드
    """
    print("한국어 금융 감성 분석 모델 로딩 중...")
    
    # snunlp/KR-FinBert-SC 가 금융 특화라 가장 추천됨
    model_candidates = [
        "snunlp/KR-FinBert-SC",  
        "monologg/kofinbert",
        "beomi/KcELECTRA-base",
        "ProsusAI/finbert", 
    ]
    
    tokenizer = None
    model = None
    
    for model_name in model_candidates:
        try:
            print(f"  시도 중: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model.eval()
            print(f"✅ 모델 로딩 완료: {model_name}")
            return tokenizer, model
        except Exception as e:
            print(f"  ⚠️ {model_name} 로딩 실패: {e}")
            continue
    
    print("⚠️ 모든 모델 로딩 실패. 키워드 기반 감성 분석을 사용합니다.")
    return None, None

def analyze_sentiment_keyword_based(text):
    """
    키워드 기반 감성 분석 (모델 로드 실패 시 백업용)
    """
    if pd.isna(text) or text == '' or text is None:
        return 'Neutral', 0.0, 0.0, 1.0
    
    text_lower = str(text).lower()
    
    positive_keywords = [
        '상승', '증가', '성장', '호조', '개선', '확대', '상향', '강세', '급등', '반등',
        '수익', '이익', '실적', '호재', '긍정', '낙관', '기대', '돌파', '신고가', '최고가', 
        '최대', '기록', '달성', '투자', '확장', '진출', '공급', '수요'
    ]
    
    negative_keywords = [
        '하락', '감소', '축소', '악화', '하향', '약세', '급락', '폭락', '추락',
        '손실', '손해', '부진', '악재', '부정', '비관', '우려', '하회', '미달', 
        '부족', '위축', '후퇴', '퇴보', '경고', '위험', '리스크', '불안'
    ]
    
    pos_count = sum(1 for keyword in positive_keywords if keyword in text_lower)
    neg_count = sum(1 for keyword in negative_keywords if keyword in text_lower)
    
    total_keywords = pos_count + neg_count
    if total_keywords == 0:
        return 'Neutral', 0.0, 0.0, 1.0
    
    pos_prob = min(0.9, 0.5 + (pos_count / max(total_keywords, 1)) * 0.4)
    neg_prob = min(0.9, 0.5 + (neg_count / max(total_keywords, 1)) * 0.4)
    neu_prob = 1.0 - pos_prob - neg_prob
    
    total = pos_prob + neg_prob + neu_prob
    pos_prob /= total
    neg_prob /= total
    neu_prob /= total
    
    if pos_prob > neg_prob and pos_prob > neu_prob:
        sentiment = 'Positive'
    elif neg_prob > pos_prob and neg_prob > neu_prob:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment, pos_prob, neg_prob, neu_prob

def analyze_sentiment_batch(texts, tokenizer, model, device='cpu', batch_size=32):
    """
    배치 단위 감성 분석 (속도 최적화 & 라벨 자동 매핑)
    """
    results = {'sentiment': [], 'positive_prob': [], 'negative_prob': [], 'neutral_prob': []}
    clean_texts = [str(t) if pd.notna(t) and t != '' else '' for t in texts]
    id2label = model.config.id2label
    num_batches = math.ceil(len(clean_texts) / batch_size)
    
    for i in tqdm(range(num_batches), desc="   딥러닝 분석 중", leave=False):
        batch_texts = clean_texts[i*batch_size : (i+1)*batch_size]
        
        if all(t == '' for t in batch_texts):
            for _ in batch_texts:
                results['sentiment'].append('Neutral')
                results['positive_prob'].append(0.0)
                results['negative_prob'].append(0.0)
                results['neutral_prob'].append(1.0)
            continue

        try:
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
            if device == 'cuda':
                inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            for j, prob_arr in enumerate(probs):
                if batch_texts[j] == '':
                    results['sentiment'].append('Neutral')
                    results['positive_prob'].append(0.0)
                    results['negative_prob'].append(0.0)
                    results['neutral_prob'].append(1.0)
                    continue

                score_map = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
                if id2label:
                    for idx, score in enumerate(prob_arr):
                        label_name = str(id2label[idx]).lower()
                        if 'pos' in label_name: score_map['positive'] = float(score)
                        elif 'neg' in label_name: score_map['negative'] = float(score)
                        elif 'neu' in label_name: score_map['neutral'] = float(score)
                        elif "finbert" in model.name_or_path.lower(): # snunlp 예외처리
                            if idx == 0: score_map['negative'] = float(score)
                            elif idx == 1: score_map['neutral'] = float(score)
                            elif idx == 2: score_map['positive'] = float(score)
                else:
                    if len(prob_arr) == 3:
                        score_map['negative'] = float(prob_arr[0])
                        score_map['neutral'] = float(prob_arr[1])
                        score_map['positive'] = float(prob_arr[2])
                    elif len(prob_arr) == 2:
                        score_map['positive'] = float(prob_arr[0])
                        score_map['negative'] = float(prob_arr[1])

                best_label = max(score_map, key=score_map.get).capitalize()
                results['sentiment'].append(best_label)
                results['positive_prob'].append(score_map['positive'])
                results['negative_prob'].append(score_map['negative'])
                results['neutral_prob'].append(score_map['neutral'])
                
        except Exception as e:
            print(f"Batch Error: {e}")
            for _ in batch_texts:
                results['sentiment'].append('Neutral')
                results['positive_prob'].append(0.0)
                results['negative_prob'].append(0.0)
                results['neutral_prob'].append(1.0)

    return results

def calculate_effective_date(news_date, news_time=None):
    """
    15:25 이후 뉴스는 다음 영업일로 처리
    """
    if isinstance(news_date, str):
        try:
            if news_time and pd.notna(news_time): news_datetime = pd.to_datetime(f"{news_date} {news_time}")
            else: news_datetime = pd.to_datetime(news_date)
        except: news_datetime = pd.to_datetime(news_date)
    elif isinstance(news_date, pd.Timestamp): news_datetime = news_date.to_pydatetime()
    elif isinstance(news_date, datetime): news_datetime = news_date
    elif hasattr(news_date, 'date'): news_datetime = datetime.combine(news_date, datetime.min.time())
    else: news_datetime = pd.to_datetime(news_date)
    
    if isinstance(news_datetime, datetime):
        if not hasattr(news_datetime, 'hour') or news_datetime.hour is None:
            news_datetime = news_datetime.replace(hour=12, minute=0, second=0)
    else:
        if pd.isna(news_datetime.hour) if hasattr(news_datetime, 'hour') else True:
            news_datetime = news_datetime.replace(hour=12, minute=0, second=0)
        news_datetime = news_datetime.to_pydatetime()
    
    market_close_hour, market_close_minute = 15, 25
    weekday = news_datetime.weekday()
    news_time = news_datetime.time()
    close_time = datetime.min.replace(hour=market_close_hour, minute=market_close_minute).time()
    
    if weekday >= 5: 
        effective_date = (news_datetime + timedelta(days=(7 - weekday))).date()
    elif news_time >= close_time:
        next_day = news_datetime + timedelta(days=1)
        if next_day.weekday() >= 5:
            effective_date = (next_day + timedelta(days=(7 - next_day.weekday()))).date()
        else:
            effective_date = next_day.date()
    else:
        effective_date = news_datetime.date()
    
    return effective_date

def process_news_files(data_dir, output_dir=None, theme_keywords=None):
    """
    theme_keywords: 필터링할 주제어 리스트 (None이면 필터링 안 함)
    """
    tokenizer, model = load_sentiment_model()
    use_keyword_based = (tokenizer is None or model is None)
    
    device = 'cpu'
    if not use_keyword_based and torch.cuda.is_available():
        device = 'cuda'
        model = model.to(device)
        print(f"사용 디바이스: {device} (GPU 가속 활성화)")
    
    if output_dir is None: output_dir = data_dir
    
    news_files = []
    if os.path.isdir(data_dir):
        for file in os.listdir(data_dir):
            if file.startswith('NewsResult_with_sentiment_'): continue
            if file.endswith(('.xlsx', '.xls', '.csv')) and 'News' in file:
                news_files.append(os.path.join(data_dir, file))
    else:
        if not os.path.basename(data_dir).startswith('NewsResult_with_sentiment_'):
            news_files = [data_dir]
    
    if len(news_files) == 0:
        print("⚠️ 처리할 파일이 없습니다.")
        return []
    
    all_results = []
    
    for file_path in news_files:
        print(f"\n{'='*50}")
        print(f"📄 파일 로드: {os.path.basename(file_path)}")
        try:
            if file_path.endswith('.csv'): df = pd.read_csv(file_path)
            else: df = pd.read_excel(file_path)
            
            # --- 텍스트 컬럼 탐색 ---
            exclude_keywords = ['식별자', 'id', 'identifier', '번호', 'number', '코드', 'code']
            text_columns = []
            for col in df.columns:
                col_lower = str(col).lower()
                if any(k in col_lower for k in ['title', '제목', 'content', '본문']):
                    if not any(ek in col_lower for ek in exclude_keywords): text_columns.append(col)
            
            if not text_columns:
                for col in df.columns:
                    if df[col].dtype == 'object':
                        if len(str(df[col].iloc[0])) > 30: text_columns.append(col)

            # 텍스트 결합
            if len(text_columns) == 1:
                df['combined_text'] = df[text_columns[0]].fillna('').astype(str)
            else:
                df['combined_text'] = df[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
            
            original_count = len(df)
            
            # --- [핵심] 테마 키워드 필터링 ---
            if theme_keywords and len(theme_keywords) > 0:
                print(f"🔍 테마 필터링 적용 중... (키워드: {theme_keywords})")
                
                # 키워드가 하나라도 포함되면 True
                # (대소문자 구분 없이 검색하기 위해 lower() 적용)
                def check_keywords(text):
                    text_lower = str(text).lower()
                    return any(k.lower() in text_lower for k in theme_keywords)
                
                mask = df['combined_text'].apply(check_keywords)
                df = df[mask].reset_index(drop=True)
                
                filtered_count = len(df)
                removed_count = original_count - filtered_count
                print(f"   📉 {original_count}건 -> {filtered_count}건 (테마와 무관한 {removed_count}건 제외됨)")
                
                if filtered_count == 0:
                    print("   ⚠️ 필터링 결과 남은 뉴스가 없습니다. 다음 파일로 넘어갑니다.")
                    continue
            else:
                print("   ➡️ 테마 필터링 미적용 (모든 뉴스 분석)")

            # --- 감성 분석 실행 ---
            if use_keyword_based:
                sentiments, p_probs, n_probs, neu_probs = [], [], [], []
                for txt in tqdm(df['combined_text'], desc="   키워드 분석 중"):
                    s, p, n, nu = analyze_sentiment_keyword_based(txt)
                    sentiments.append(s); p_probs.append(p); n_probs.append(n); neu_probs.append(nu)
                df['Sentiment'] = sentiments; df['Positive_Prob'] = p_probs
                df['Negative_Prob'] = n_probs; df['Neutral_Prob'] = neu_probs
            else:
                batch_res = analyze_sentiment_batch(df['combined_text'].tolist(), tokenizer, model, device, batch_size=32)
                df['Sentiment'] = batch_res['sentiment']
                df['Positive_Prob'] = batch_res['positive_prob']
                df['Negative_Prob'] = batch_res['negative_prob']
                df['Neutral_Prob'] = batch_res['neutral_prob']

            df['Sentiment_Score'] = df['Positive_Prob'] - df['Negative_Prob']
            
            # --- 영향일자 계산 ---
            date_col, time_col = None, None
            for col in df.columns:
                c_str = str(col).lower()
                if '일자' in c_str or 'date' in c_str: date_col = col
                if '시간' in c_str or 'time' in c_str: time_col = col
            
            if date_col:
                eff_dates = []
                for i in range(len(df)):
                    d = df[date_col].iloc[i]
                    t = df[time_col].iloc[i] if time_col else None
                    eff_dates.append(calculate_effective_date(d, t))
                df['Effective_Date'] = eff_dates
            
            # 저장
            base = os.path.basename(file_path).replace('NewsResult_with_sentiment_', 'NewsResult_')
            name, ext = os.path.splitext(base)
            out_file = os.path.join(output_dir, f"NewsResult_with_sentiment_{name}{ext}")
            
            if file_path.endswith('.csv'): df.to_csv(out_file, index=False, encoding='utf-8-sig')
            else: df.to_excel(out_file, index=False, engine='openpyxl')
            
            print(f"   ✅ 저장 완료: {out_file}")
            all_results.append(df)
            
        except Exception as e:
            print(f"   ❌ 오류 발생: {e}")
            import traceback; traceback.print_exc()

    return all_results

# ---------------------------------------------------------
# 실행 설정
# ---------------------------------------------------------
if __name__ == "__main__":
    news_data_dir = "/home/jhkim/01_dev/03_stock_market_price_expectation/_data/01_news"
    output_dir = "/home/jhkim/01_dev/03_stock_market_price_expectation/_data/03_refined_news"
    
    # -----------------------------------------------------
    # [형님, 여기만 수정하면 돼!]
    # 분석하고 싶은 테마(관심 분야)의 키워드를 입력해줘.
    # 예: ['AI', '인공지능', '반도체', '삼성전자']
    # 만약 모든 뉴스를 다 보고 싶으면 리스트를 비워둬: []
    # -----------------------------------------------------
    # -----------------------------------------------------
    # [AI & 미래산업 테마 키워드 100선]
    # -----------------------------------------------------
    THEME_KEYWORDS = [
        # 1. 핵심 키워드 (AI & 소프트웨어)
        'AI', '인공지능', '생성형', '생성형AI', 'GenAI', 
        '챗봇', 'Chatbot', 'ChatGPT', 'GPT', 'LLM', '거대언어모델',
        '머신러닝', '딥러닝', '알고리즘', '빅데이터', '데이터센터', 
        '클라우드', 'Cloud', 'SaaS', 'PaaS', 'API',
        '신경망', 'NPU', '비전', '음성인식', '자연어처리', 'NLP',
        
        # 2. 국내 대표 플랫폼 & AI 대장주
        '네이버', 'NAVER', '하이퍼클로바', 'HyperClova', '치지직',
        '카카오', 'Kakao', '카카오브레인', 'KoGPT', 
        '삼성에스디에스', 'LG CNS', 'SK C&C', 
        
        # 3. AI 반도체 & 하드웨어 (가장 중요)
        '삼성전자', 'SK하이닉스', 'HBM', 'HBM3', 'HBM3E', 'CXL', 'PIM',
        '반도체', '메모리', '시스템반도체', '파운드리', '패키징',
        'GPU', '엔비디아', 'NVIDIA', 'AMD', '인텔', 'ARM',
        '온디바이스', 'On-Device', '엣지컴퓨팅', '스냅드래곤', '엑시노스',
        '한미반도체', 'HPSP', '이수페타시스', '리노공업', '고영', '주성엔지니어링',
        
        # 4. 로봇 & 자율주행 (AI의 손과 발)
        '로봇', 'Robot', '휴머노이드', '협동로봇', '산업용로봇', 
        '레인보우로보틱스', '두산로보틱스', '유진로봇', '티로보틱스',
        '자율주행', '모빌리티', 'SDV', '현대오토에버', '스마트카',
        '현대차', '기아', '테슬라', 'Tesla',
        
        # 5. 의료 AI & 바이오 (핫한 섹터)
        '의료AI', '디지털헬스케어', '신약개발', '유전체',
        '루닛', '뷰노', '제이엘케이', '딥노이드', 
        
        # 6. 메타버스 & 통신 & 보안
        '메타버스', 'XR', 'VR', 'AR', '디지털트윈',
        '통신', '5G', '6G', 'SK텔레콤', 'KT', 'LG유플러스',
        '보안', '사이버보안', '정보보안', '안랩', '샌즈랩', '모니터랩',
        
        # 7. AI 관련 글로벌 빅테크 (국내 뉴스에 자주 언급됨)
        '마이크로소프트', 'Microsoft', 'MS', '오픈AI', 'OpenAI',
        '구글', 'Google', '제미나이', 'Gemini', 
        '애플', 'Apple', '비전프로', '아마존', 'AWS', '메타', 'Meta',
        
        # 8. 기타 관련 주요 용어
        '스마트팩토리', '공정자동화', '수율', '공급망', '데이터댐',
        '디지털전환', 'DX', '핀테크', 'STO', '블록체인'
    ]
    
    print("="*60)
    print("뉴스 감성 분석 시작 (테마 필터링 포함)")
    print("="*60)
    
    results = process_news_files(news_data_dir, output_dir, theme_keywords=THEME_KEYWORDS)
    
    print("\n" + "="*60)
    print("✅ 모든 작업 완료!")
    print("="*60)