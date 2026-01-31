import requests
import json
import os
import time
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 1. 환경 설정
APP_KEY = os.getenv("REAL_APP_KEY")
APP_SECRET = os.getenv("REAL_APP_SECRET")
BASE_URL = "https://openapi.koreainvestment.com:9443"
TOKEN_CACHE_FILE = "D:/stock/token_cache.json"

def get_access_token():
    """
    저장된 토큰이 있고 유효하다면 불러오고, 
    그렇지 않으면 새로 발급받아 저장합니다.
    """
    
    # [Step 1] 기존에 저장된 토큰이 있는지 확인
    if os.path.exists(TOKEN_CACHE_FILE):
        with open(TOKEN_CACHE_FILE, "r") as f:
            try:
                cached_data = json.load(f)
                # 현재 시간이 만료 예정 시간보다 이전인지 확인 (여유있게 10분 전으로 설정)
                if time.time() < cached_data.get("expiry_time", 0) - 600:
                    print("✅ 유효한 기존 토큰을 로컬에서 불러왔습니다.")
                    return cached_data.get("access_token")
            except (json.JSONDecodeError, KeyError):
                print("⚠️ 토큰 캐시 파일이 손상되었습니다. 새로 발급합니다.")

    # [Step 2] 토큰 새로 발급받기
    print("🆕 토큰이 만료되었거나 존재하지 않습니다. 새로 발급을 진행합니다...")
    path = "/oauth2/tokenP"
    url = f"{BASE_URL}{path}"
    
    data = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        res_data = response.json()
        
        access_token = res_data.get("access_token")
        expires_in = int(res_data.get("expires_in", 86400)) # 기본 24시간
        
        if access_token:
            # [Step 3] 새 토큰과 만료 시간 저장 (현재 시간 + 유효 시간)
            expiry_time = time.time() + expires_in
            cache_info = {
                "access_token": access_token,
                "expiry_time": expiry_time
            }
            
            with open(TOKEN_CACHE_FILE, "w") as f:
                json.dump(cache_info, f)
            
            print(f"✅ 새 토큰 발급 및 저장 완료! (유효기간: {expires_in // 3600}시간)")
            return access_token
        else:
            print("❌ 토큰 발급 실패: 응답 데이터에 토큰이 없습니다.")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ API 요청 중 오류 발생: {e}")
        return None

# --- 실행부 ---
if __name__ == "__main__":
    token = get_access_token()
    if token:
        print(f"🚀 현재 사용 가능한 토큰: {token[:20]}...")