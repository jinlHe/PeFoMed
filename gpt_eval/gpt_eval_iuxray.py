import re
from openai import OpenAI
import json
import os
import argparse
from tqdm import tqdm

openai_key='sk-XXXXXXXXXXXXXXXXXXXXXXXX'
client = OpenAI(api_key=openai_key)

###################################################################################################iuxray
# 功能：以jsonl格式逐条写入结果
def write_result_to_jsonl(file, result):
    file.write(json.dumps(result, ensure_ascii=False) + '\n')

# 功能：更新统计信息文件
def update_statistics_file(stats_path, stats):
    with open(stats_path, 'w', encoding='utf-8') as file:
        json.dump(stats, file, ensure_ascii=False, indent=4)

resume_from_id = "CXR692_IM-2258/0.png"

base_path = "/content/drive/MyDrive/Colab_Notebooks/iuxray"
answer_files = {
    "iuxray": "iuxray.json"
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
        if resume_from_id == "-1" or not os.path.exists(stats_path):
            stats = {
                "count_0": 0,
                "count_1": 0,
                "count_2": 0,
                "count_3": 0,
                "count_4": 0,
            }
            resume_from_id = "-1"
        else:
            with open(stats_path, 'r', encoding='utf-8') as stats_file:
                stats = json.load(stats_file)

        with open(output_dir, 'a', encoding='utf-8') as results_file:
            # 设置一个标志，用来表示是否找到了恢复点
            found_resume_point = (resume_from_id == "-1")
            for item in tqdm(data):
                if not found_resume_point:
                  if item['image_path'] != resume_from_id:
                    continue  # 跳过，直到找到恢复点
                  else:
                  # 找到恢复点，标记并在下一轮开始处理
                    found_resume_point = True
                    continue  # 避免重复处理恢复点，直接进入下一个循环

                image_path = item['image_path']
                ground_truth = item['ground_truth']
                generated_report = item['generated_report']

                prompt = f"""
                Given a actual medical report on a medical image and a generated medical report that needs to be determined, evaluate the generated report to be determined (0 or 1 or 2 or 3 or 4).

                standard :
                4: The generated medical report exhibits minimal discrepancy in meaning compared to the actual  report, ensuring complete consistency in critical aspects such as diagnosis and treatment recommendations. It encapsulates all requisite information elements without omissions.
                Applicability: This level is pertinent when evaluators face difficulty distinguishing between the generated and actual medical reports.
                3: Generated medical reports closely resemble actual reports in critical elements, albeit with minor deviations that do not compromise the overall accuracy and intelligibility. They maintain alignment with the principal diagnosis and recommendations of the authentic reports, despite possible negligible discrepancies in non-essential information. These reports encompass nearly all vital information, with minimal omissions that do not impede comprehensive understanding.
                Applicability: This criterion applies when the generated report's overall quality approximates that of the actual report, albeit with slight variances in detail.
                2: The generated report diverges significantly from the actual report in several key aspects, yet its primary content and intent remain recognizable. Despite discrepancies in crucial information, the core diagnosis and recommendations are accurate. Omissions of some information elements have a marginal impact on the report's overall comprehensibility.
                Applicability: This level is relevant when the report's foundational structure mirrors that of the actual report, despite a notable quantity and severity of inaccuracies or omissions.
                1: The generated report is quite different from the actual report in several key aspects, which affects the basic understanding and use of the reports. There are multiple errors or misleading statements of key information. More important information is left out, affecting the completeness and practicality of the report.
                Applicability: Although the report retains some basic framework or content, there are many errors, which affect the overall quality.
                0: The generated report hardly has any consistency with the actual report and completely deviates from the correct message or purpose. The information in the report is completely inconsistent with the actual report and is full of errors or irrelevant content. It lacks the necessary information elements to serve as an effective medical report.
                Applicability: It is applicable when the content of the report is completely inconsistent with the intended goal and it is almost impossible to identify it as a valid medical report.

                actual medical report:
                - actual medical report: {ground_truth}\n

               generated medical report:
               - generated medical report to be determined: {generated_report}\n

                Task:\n
                - Given a actual medical report on a medical image and a generated medical report that needs to be determined, evaluate the generated report to be determined (0 or 1 or 2 or 3 or 4).

                Output Format(Only "Score:" and a rating number can be output):
                Score: your answer\n
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
                    match = re.search(r'\b[0-5](?:\.\d+)?\b', gpt_response.split('Score:')[1])
                    # print(f"match: {match}")
                    evaluation_score = int(match.group(0))
                    # print(f"score: {evaluation_score}")

                    if evaluation_score == 0:
                      stats["count_0"] += 1
                    elif evaluation_score == 1:
                      stats["count_1"] += 1
                    elif evaluation_score == 2:
                      stats["count_2"] += 1
                    elif evaluation_score == 3:
                      stats["count_3"] += 1
                    elif evaluation_score == 4:
                      stats["count_4"] += 1
                    else:
                      print(f"not 0 to 4:'{evaluation_score}'.")
                    result ={
                        "id": image_path,
                        "ground_truth": ground_truth,
                        "generated_report": generated_report,
                        "evaluation_score": evaluation_score
                    }
                    write_result_to_jsonl(results_file, result)
                    # 更新并保存统计信息
                    update_statistics_file(stats_path, stats)
                except:
                    print(f"Error sample ID '{key}', GPT response: '{gpt_response}'.")