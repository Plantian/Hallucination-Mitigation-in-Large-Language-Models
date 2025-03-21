import json
import requests

def generate_response(question):
    """
    使用 Ollama 的 API 获取 deepseek-r1:8b 模型对问题的响应
    """
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "deepseek-r1:7b",
        "prompt": question,
        "temperature": 0.7,
        "max_tokens": 500
    }
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if line:
                line_data = line.decode("utf-8")
                try:
                    json_data = json.loads(line_data)
                    if "response" in json_data:
                        full_response += json_data["response"]

                    if json_data.get("done", False):
                        break
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line for question '{question}': {e}")
                    print(f"Line content: {line_data}")
                    return None
        
        return full_response
    
    except Exception as e:
        print(f"Error generating response for question '{question}': {e}")
        return None

def main():
    with open("HalluQA.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    responses = []
    for item in data:
        question = item.get("Question", "")
        if not question:
            print(f"Skipping item with no question: {item}")
            continue
        response = generate_response(question)
        if response:
            print(response) # this is monitors
            responses.append({
                "question": question,
                "response": response
            })

    with open("deepseek_responses.json", "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)
    
    print("Response file generated successfully!")

if __name__ == "__main__":
    main()