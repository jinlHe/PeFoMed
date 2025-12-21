"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from PIL import Image
import os
import json
import random

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image_name"],
                "question": ann["question"],
                "question_id": ann["qid"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image_name"],
            }
        )


class MEDVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, use_instruction_pool=True):
        """
        初始化医学VQA数据集
        
        参数:
            vis_processor: 视觉处理器
            text_processor: 文本处理器
            vis_root: 图像根目录
            ann_paths: 标注文件路径
            use_instruction_pool: 是否使用instruction pool
                - True: 使用完整的instruction pool (适用于LoRA等方法)
                - False: 使用简化的task prefix (适用于Prompt Tuning)
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.use_instruction_pool = use_instruction_pool
        
        if use_instruction_pool:
            # 完整的instruction pool，适用于LoRA/Full Fine-tuning
            self.instruction_pool = [
                "[vqa] {}",
                "[vqa] Based on the image, respond to this question with a short answer: {}"
            ]
        else:
            # 简化的task prefix，适用于Prompt Tuning
            # Prompt Tuning的虚拟tokens会学习详细的指令表示
            self.task_prefix = "[vqa]"

        exist_annotation = []
        for ann in self.annotation:
            image_path = os.path.join(self.vis_root, ann["image_name"])
            if os.path.exists(image_path):
                exist_annotation.append(ann)
        self.annotation = exist_annotation


    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image_name"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        answers = str(ann["answer"]) if isinstance(ann["answer"], int) else ann["answer"]
        answers = self.text_processor(answers)

        # 根据配置选择instruction生成方式
        if self.use_instruction_pool:
            # LoRA/Full FT: 使用随机的详细instruction
            instruction_quetion = random.choice(self.instruction_pool).format(ann['question'])
        else:
            # Prompt Tuning: 使用简化的task prefix + 问题
            instruction_quetion = f"{self.task_prefix} {ann['question']}"
        
        instruction_quetion = "<Img><ImageHere></Img> {} ".format(instruction_quetion)
        instruction_quetion = self.text_processor(instruction_quetion)

        weights = [1.]

        return {
            "image": image,
            "text_input": instruction_quetion,
            "answers": answers,
            "weights": weights
        }

class MEDVQAEvalData(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, use_instruction_pool=True):
        """
        初始化医学VQA评估数据集
        
        参数:
            vis_processor: 视觉处理器
            text_processor: 文本处理器
            vis_root: 图像根目录
            ann_paths: 标注文件路径
            use_instruction_pool: 是否使用详细instruction
                - True: 使用详细instruction (与训练时的instruction pool保持一致)
                - False: 使用简化的task prefix (适用于Prompt Tuning)
        """
        self.vis_root = vis_root
        self.annotation = json.load(open(ann_paths[0]))
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.use_instruction_pool = use_instruction_pool

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image_name"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = ann["question"]
        
        # 根据配置选择instruction格式（评估时保持与训练一致）
        if self.use_instruction_pool:
            # 使用详细instruction（与训练时的第二个模板保持一致）
            question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        else:
            # Prompt Tuning: 使用简化的task prefix
            question = f"[vqa] {question}"
        
        question = self.text_processor(question)

        return {
            "image": image,
            "text_input": question,
            "qid": ann["qid"]
        }

