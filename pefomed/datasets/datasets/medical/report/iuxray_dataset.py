import os
from PIL import Image
import random
from pefomed.datasets.datasets.caption_datasets import CaptionDataset, CaptionInstructDataset, CaptionEvalDataset
from pefomed.datasets.datasets.base_dataset import BaseDataset
from collections import OrderedDict

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image_path"],
                "report": ann["report"],
                "image": sample["image"],
            }
        )


class IUXRAYDataset(CaptionDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_path"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
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
        ann = self.annotation[index]

        img_file = ann["image_path"]
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["report"])
        # print("image_path:", image_path)
        # print("report:", caption)

        instruction = random.choice(self.instruction_pool)
        instruction = self.text_processor("<Img><ImageHere></Img> [report] {} ".format(instruction))

        return {
            "image": image,
            "text_input": instruction,
            "answer": caption
        }


class IUXRAYEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

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
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image_path"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        instruction = random.choice(self.instruction_pool)
        question = self.text_processor("[caption] {}".format(instruction))
        caption = self.text_processor(ann["report"])
        return {
            "caption": caption,
            "text_input": question,
            "image": image,
            "image_id": ann["image_path"],
            "instance_id": ann["image_path"],
        }
