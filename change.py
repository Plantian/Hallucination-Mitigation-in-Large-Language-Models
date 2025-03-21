import json
'''
add options: question_id to the items in json file
'''
# input file path
input_file = "deepseek_responses.json"
# output files path
output_file = "deepseek_responses1.json"
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)
for index, item in enumerate(data, start=1):
    item["question_id"] = index
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"McYep {output_file}")