import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. 모델 및 토크나이저 로드
model_name = "clare25/krfinbert-jongtobang"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_sentiment_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    probs = F.softmax(logits, dim=-1).squeeze().tolist()
    
    # [교정] 테스트 결과에 따라 0번을 긍정(Bullish), 1번을 부정(Bearish)으로 설정
    # 점수 = P(0) - P(1) -> 긍정일 때 +1에 가까워짐
    score = probs[0] - probs[1]
    
    return score

# 테스트
news_test = "트럼프 '한국이 약속 안 지켜' 주장 사실?…'관세 25% 상향' 진짜 이유 [핫이슈]"
print(f"교정된 감성 점수: {get_sentiment_score(news_test):.4f}")