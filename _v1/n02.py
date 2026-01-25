import re

# 1. 파일에서 모든 키워드 로드
with open('keyword_keys.txt', 'r', encoding='utf-8') as f:
    all_keys = [line.strip() for line in f.readlines()]

# 2. 강력한 매칭을 위한 핵심 키워드 패턴 정의
# 삼성전자 및 반도체/모바일 핵심 생태계 키워드 중심
core_pattern = r"삼성전자|삼전|이재용|반도체|HBM|DRAM|갤럭시|파운드리|실적|영업이익|블록딜|상속세|엔비디아|TSMC|하이닉스"

# 3. 필터링 수행
# 핵심 단어를 포함하거나, '삼성'이라는 글자가 들어간 경제/IT 키워드만 추출
refined_keywords = [
    key for key in all_keys 
    if re.search(core_pattern, key) or (key.startswith('삼성') and len(key) > 2)
]

# 4. 결과 확인
print(f"전체 {len(all_keys)}개 중 {len(refined_keywords)}개의 유효 키워드 추출 완료")

# 1. 저장할 파일 경로 설정
save_path = "d:/stock/news_refiner/refined_keywords.txt"

# 2. 파일 쓰기
with open(save_path, 'w', encoding='utf-8') as f:
    for word in refined_keywords:
        f.write(word + '\n')

print(f"✅ 정제된 키워드 {len(refined_keywords)}개가 다음 경로에 저장되었습니다: {save_path}")