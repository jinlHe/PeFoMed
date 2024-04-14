import re
from openai import OpenAI
import json
import os
import argparse
from tqdm import tqdm

openai_key='sk-XXXXXXXXXXXXXXXXXXXXXXXX'
client = OpenAI(api_key=openai_key)

# 功能：以jsonl格式逐条写入结果
def write_result_to_jsonl(file, result):
    file.write(json.dumps(result, ensure_ascii=False) + '\n')

# 功能：更新统计信息文件
def update_statistics_file(stats_path, stats):
    with open(stats_path, 'w', encoding='utf-8') as file:
        json.dump(stats, file, ensure_ascii=False, indent=4)

resume_from_id = "300688007"

base_path = "/content/drive/MyDrive/Colab_Notebooks/pathvqa/MM_2024/"
answer_files = {
    "pathvqa_stage1_stage2": "pathvqa_stage1_stage2.json",
    "pathvqa_stage2": "pathvqa_stage2.json",
    "pathvqa_stage1": "pathvqa_stage1.json",
    "pathvqa": "pathvqa.json",
}

for subfolder, answer_file_name in answer_files.items():
    answer_file_path = os.path.join(base_path, subfolder, answer_file_name)
    for i in range(1, 2):
        output_filename = f"gpt4result{i}.jsonl"
        output_dir = os.path.join(base_path, subfolder, output_filename)
        stats_path = os.path.join(base_path, subfolder, f"gpt4result{i}_stats.json")
        print(f"Processing: {answer_file_path} -> {output_dir}")
        with open(answer_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            data = json.loads(content)

        # 初始化或加载统计信息
        if resume_from_id == -1 or not os.path.exists(stats_path):
            stats = {
                "total_count": 0,
                "count_Open_1": 0,
                "count_Close_1": 0,
                "count_Open_0": 0,
                "count_Close_0": 0,
                "image_organ_count_1": {},
                "image_organ_count_0": {},
                "question_type_count_1": {},
                "question_type_count_0": {},
            }
            resume_from_id = "-1"
        else:
            with open(stats_path, 'r', encoding='utf-8') as stats_file:
                stats = json.load(stats_file)

        with open(output_dir, 'a', encoding='utf-8') as results_file:
            # 设置一个标志，用来表示是否找到了恢复点
            found_resume_point = (resume_from_id == "-1")
            for key, item in tqdm(data.items()):
                if not found_resume_point:
                  if key != resume_from_id:
                    continue  # 跳过，直到找到恢复点
                  else:
                  # 找到恢复点，标记并在下一轮开始处理
                    found_resume_point = True
                    continue  # 避免重复处理恢复点，直接进入下一个循环

                stats["total_count"] += 1
                question = item['question']
                true_answer = item['ground_truth']  # Ground truth
                generated_answer = item['generated_answer']  # Answer to be determined
                answer_type = item['answer_type']  # This is either "OPEN" or "CLOSED"
                # phrase_type = item['phrase_type']
                question_type = item['question_type']

                prompt = f"""
                Given a question about an medical image, there is a correct answer to the question and an answer to be determined. If the answer to be determined matches the correct answer or is a good enough answer to the question, output 1; otherwise output 0. Evaluate the answer to be determined (1 or 0).

                Question:
                - question about the medical image: {question}\n

                Answers:
                - correct answer(ground truth): {true_answer}\n
                  answer to be determined: {generated_answer}\n

                Task:\n
                - Given a question about an medical image, there is a correct answer to the question and an answer to be determined. If the answer to be determined matches the correct answer or is a good enough answer to the question, output 1; otherwise output 0. Evaluate the answer to be determined (1 or 0).

                Output Format:
                Correctness: your answer\n
                """

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": prompt}, ]
                        }
                    ],
                    max_tokens=100,
                )

                gpt_response = response.choices[0].message.content
                # print(f"\n gpt_response: {gpt_response}")
                try:
                    match = re.search(r'\b[0-5](?:\.\d+)?\b', gpt_response.split('Correctness:')[1])
                    # print(f"match: {match}")
                    evaluation_score = int(match.group(0))
                    # print(f"score: {evaluation_score}")

                    if evaluation_score == 1:
                      if answer_type == "OPEN":
                          stats["count_Open_1"] += 1
                      else:
                          stats["count_Close_1"] += 1
                      stats["question_type_count_1"][question_type] = stats["question_type_count_1"].get(question_type, 0) + 1
                    else:
                      if answer_type == "OPEN":
                          stats["count_Open_0"] += 1
                      else:
                          stats["count_Close_0"] += 1
                      stats["question_type_count_0"][question_type] = stats["question_type_count_0"].get(question_type, 0) + 1


                    result ={
                        "id": key,
                        "question": question,
                        "correct_answer": true_answer,
                        "generated_answer": generated_answer,
                        "evaluation_score": evaluation_score
                    }
                    write_result_to_jsonl(results_file, result)
                    # 更新并保存统计信息
                    update_statistics_file(stats_path, stats)
                except:
                    print(f"Error sample ID '{key}', GPT response: '{gpt_response}'.")