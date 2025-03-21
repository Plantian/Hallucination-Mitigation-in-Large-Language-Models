import json
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--response_file_name', type=str, default='deepseek_output.json')
    return parser.parse_args()

def load_data(file_name):
    with open(file_name, 'r',encoding="UTF-8") as f:
        data = json.load(f)
    return data

def calculate_acc(predicts, ground_truth):
    correct_count = 0
    for i in range(len(predicts)):
        correct_choice = ground_truth[i]["answer"][len('Answer: '):].strip()
        response = predicts[i]['response'].strip()
        if response.startswith('Answer: '):
            if response[len('Answer: '):] == correct_choice:
                correct_count += 1
        elif len(response) == 1 and response.isalpha():
            if response == correct_choice:
                correct_count += 1
    return correct_count / len(predicts)

if __name__ == '__main__':
    args = get_args()
    predicts = load_data('deepseek7b_predictions.json')
    ground_truth = load_data('HalluQA_mc.json')
    print('Acc: {:.2f}%'.format(100 * calculate_acc(predicts, ground_truth)))