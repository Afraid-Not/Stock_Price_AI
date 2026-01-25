import requests
import os
from dotenv import load_dotenv

load_dotenv()

# 네이버 개발자 센터에서 발급받은 키
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

def get_today_naver_news(keyword="삼성전자"):
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    params = {
        "query": keyword,
        "display": 20, # 뉴스 20건
        "sort": "sim"  # 유사도순
    }

    try:
        res = requests.get(url, headers=headers, params=params)
        if res.status_code == 200:
            items = res.json().get('items', [])
            # HTML 태그 제거 및 제목 추출
            titles = [item['title'].replace('<b>', '').replace('</b>', '').replace('&quot;', '"') for item in items]
            print(f"📰 네이버에서 '{keyword}' 관련 뉴스 {len(titles)}건 수집 완료.")
            return titles
        else:
            print(f"❌ 네이버 API 실패: {res.status_code}")
            return []
    except Exception as e:
        print(f"❌ 오류: {e}")
        return []