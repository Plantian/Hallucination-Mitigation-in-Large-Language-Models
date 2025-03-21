import json
import requests
from typing import List, Dict

class OllamaProcessor:
    def __init__(self, api_url: str = "http://localhost:11434/api/generate", model: str = "deepseek-r1:7b"):
        self.api_url = api_url
        self.model = model
        self.headers = {"Content-Type": "application/json"}

    def _call_api(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "max_tokens": 100,
                "repeat_penalty": 1.1
            }
        }
        try:
            response = requests.post(self.api_url, json=payload, headers=self.headers, timeout=60)
            # you can add request time
            response.raise_for_status()
            return response.json()['response'].strip()
        except requests.exceptions.RequestException as e:
            print(f"API request false: {str(e)}")
            return "Error: API request false"
        except KeyError:
            print("encoding response failed")
            return "Error: invalid response"

    def process_questions(self, input_path: str, output_path: str = "res.json"):
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                questions = json.load(f)
        except FileNotFoundError:
            print(f"input error {input_path} don't exit")
            return
        except json.JSONDecodeError:
            print("invalid input")
            return

        results = []
        indexs = 1
        for item in questions:
            response = self._call_api(item["question"])
            print(indexs)
            indexs += 1
            results.append({
                "question_id": item["question_id"],
                "question": item["question"],
                "response": f"Answer: {response}"
            })

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"McYep done! {output_path}")


if __name__ == "__main__":
    processor = OllamaProcessor()

    processor.process_questions(
        input_path="HalluQA_mc.json",
        output_path="deepseek_responses_mc.json"
    )