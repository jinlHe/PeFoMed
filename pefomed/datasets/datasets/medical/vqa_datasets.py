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

from pefomed.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
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
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool =[
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

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

        instruction_quetion = random.choice(self.instruction_pool).format(ann['question'])
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
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        self.vis_root = vis_root
        self.annotation = json.load(open(ann_paths[0]))
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image_name"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = ann["question"]
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        question = self.text_processor(question)

        return {
            "image": image,
            "text_input": question,
            "qid": ann["qid"]
        }

