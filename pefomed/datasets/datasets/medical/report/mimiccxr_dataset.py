import os
from PIL import Image

from pefomed.datasets.datasets.caption_datasets import CaptionDataset, CaptionInstructDataset, CaptionEvalDataset
from pefomed.datasets.datasets.base_dataset import BaseDataset
from collections import OrderedDict
import random
import torch


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "report": ann["caption"],
                "image": sample["image"],
            }
        )


class MIMICCXRDataset(CaptionDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        # 将字典键存储在列表中
        self.image_keys = list(self.annotation.keys())
        # print("image_keys1:", self.image_keys)

        exist_annotation = {}
        for img_key in self.image_keys:
            image_path = os.path.join(self.vis_root, img_key)
            if os.path.exists(image_path):
                exist_annotation[img_key] = self.annotation[img_key]
        self.annotation = exist_annotation

        self.image_keys = list(self.annotation.keys())
        # print("image_keys2:", self.image_keys)

        self.img_ids = {}
        n = 0
        for key in self.image_keys:
            if key not in self.img_ids.keys():
                self.img_ids[key] = n
                n += 1

        self.instruction_pool = [
            'Describe the given chest x-ray image in detail.',
            'Take a look at this chest x-ray and describe the findings and impression.',
            'Could you provide a detailed description of the given x-ray image?',
            'Describe the given chest x-ray image as detailed as possible.',
            'What are the key findings in this chest x-ray image?',
            'Could you highlight any abnormalities or concerns in this chest x-ray image?',
            'What specific features of the lungs and heart are visible in this chest x-ray image?',
            'What is the most prominent feature visible in this chest x-ray image, and how is it indicative of the patient\'s health?',
            'What are the finding and overall impression provided by this chest x-ray image?',
            'How does the size and shape of the heart in this chest x-ray image?',
            'Is the overall impression provided by this chest x-ray image normal or abnormal? Answer based on the observed findings.',
            'Are there any indications of infection or inflammation in this chest x-ray image, and if so, what is the likely cause?',
            'Based on the findings in this chest x-ray image, what is the overall impression?',
            'Are there any visible indications of enlargement or abnormalities in the patient\'s lymph nodes in this chest x-ray image?',
            'Is there any potential complications or risks are associated with the observed abnormalities in this chest x-ray image? or the x-ray is normal?',
        ]


    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        img_key = self.image_keys[index]
        # print("img_key3:", img_key)
        caption = self.annotation[img_key][0]
        # print("caption:", caption)
        caption = self.text_processor(caption)
        # print("filename:", filename)

        image_path = os.path.join(self.vis_root, img_key)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        # print("image_path:", image_path)
        # print("report:", caption)

        instruction = random.choice(self.instruction_pool)
        instruction = self.text_processor("<Img><ImageHere></Img> [caption] {} ".format(instruction))

        return {
            "image": image,
            "answer": caption,
            "text_input": instruction,
        }


class MIMICCXREvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        self.image_keys = list(self.annotation.keys())

        self.instruction_pool = [
            'Describe the given chest x-ray image in detail.',
            'Take a look at this chest x-ray and describe the findings and impression.',
            'Could you provide a detailed description of the given x-ray image?',
            'Describe the given chest x-ray image as detailed as possible.',
            'What are the key findings in this chest x-ray image?',
            'Could you highlight any abnormalities or concerns in this chest x-ray image?',
            'What specific features of the lungs and heart are visible in this chest x-ray image?',
            'What is the most prominent feature visible in this chest x-ray image, and how is it indicative of the patient\'s health?',
            'What are the finding and overall impression provided by this chest x-ray image?',
            'How does the size and shape of the heart in this chest x-ray image?',
            'Is the overall impression provided by this chest x-ray image normal or abnormal? Answer based on the observed findings.',
            'Are there any indications of infection or inflammation in this chest x-ray image, and if so, what is the likely cause?',
            'Based on the findings in this chest x-ray image, what is the overall impression?',
            'Are there any visible indications of enlargement or abnormalities in the patient\'s lymph nodes in this chest x-ray image?',
            'Is there any potential complications or risks are associated with the observed abnormalities in this chest x-ray image? or the x-ray is normal?',
        ]

    def __getitem__(self, index):
        # 使用索引获取图像键
        img_key = self.image_keys[index]
        # print("filename:", filename)
        image_path = os.path.join(self.vis_root, img_key)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        instruction = random.choice(self.instruction_pool)
        question = self.text_processor("[caption] {}".format(instruction))

        caption = self.annotation[img_key][0]
        caption = self.text_processor(caption)

        return {
            "caption": caption,
            "text_input": question,
            "image": image,
            "image_id": img_key,
            "instance_id": img_key,
        }
