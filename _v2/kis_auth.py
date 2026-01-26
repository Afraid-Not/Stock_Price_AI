import requests
import json
import os
import time
from dotenv import load_dotenv

class KISAuth:
    def __init__(self):
        load_dotenv()
        self.APP_KEY = os.getenv("REAL_APP_KEY")
        self.APP_SECRET = os.getenv("REAL_APP_SECRET")
        self.BASE_URL = "https://openapi.koreainvestment.com:9443"
        self.TOKEN_CACHE_FILE = "token_cache.json"

    def get_access_token(self):
        if os.path.exists(self.TOKEN_CACHE_FILE):
            with open(self.TOKEN_CACHE_FILE, "r") as f:
                try:
                    cached_data = json.load(f)
                    if time.time() < cached_data.get("expiry_time", 0) - 600:
                        return cached_data.get("access_token")
                except (json.JSONDecodeError, KeyError):
                    pass

        url = f"{self.BASE_URL}/oauth2/tokenP"
        data = {"grant_type": "client_credentials", "appkey": self.APP_KEY, "appsecret": self.APP_SECRET}
        res = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))
        res.raise_for_status()
        res_data = res.json()
        
        access_token = res_data.get("access_token")
        expiry_time = time.time() + int(res_data.get("expires_in", 86400))
        
        with open(self.TOKEN_CACHE_FILE, "w") as f:
            json.dump({"access_token": access_token, "expiry_time": expiry_time}, f)
        return access_token