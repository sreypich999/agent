import os
import requests
from dotenv import load_dotenv

load_dotenv()

class DeepSeekAdapter:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY in environment.")

    def generate_response(self, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "DeepSeek-V3-Base",
            "messages": [
                {"role": "system", "content": "You are a document analysis assistant. Answer strictly using provided context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
