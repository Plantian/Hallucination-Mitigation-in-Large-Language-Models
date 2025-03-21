import json
import requests

def load_data(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_predictions(input_file, output_file):
    questions = load_data(input_file)
    predictions = []
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20)
    session.mount('http://', adapter)
    total_questions = len(questions)
    for i, question in enumerate(questions, start=1):
        print(f"正在处理问题 {i}/{total_questions}")
        question_text = question["question"]
        prompt = f"请回答以下问题：{question_text}\n请从选项中选择一个字母作为答案，例如：Answer: A"
        payload = {
            "model": "deepseek-r1:8b",
            "prompt": prompt,
            "stream": False
        }
        try:
            response = session.post("http://localhost:11434/api/generate", json=payload)
            response.raise_for_status()

            model_response = response.json()["response"]
            print("Response successfully!")
            if "Answer:" in model_response:
                answer_part = model_response.split("Answer:")[-1].strip()
                if answer_part and answer_part[0].isalpha():
                    answer = answer_part[0].upper()
                else:
                    answer = ""
            else:
                stripped_response = model_response.strip()
                if len(stripped_response) == 1 and stripped_response.isalpha():
                    answer = stripped_response.upper()
                else:
                    answer = ""

            prediction = {
                "question_id": question["question_id"],
                "response": f"Answer: {answer}"
            }
            predictions.append(prediction)
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            prediction = {
                "question_id": question["question_id"],
                "response": ""
            }
            predictions.append(prediction)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    
    print(f"预测结果已保存到 {output_file}")

if __name__ == '__main__':
    input_file = 'HalluQA_mc.json'
    output_file = 'deepseek7b_predictions.json'
    generate_predictions(input_file, output_file)