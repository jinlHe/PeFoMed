"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from pefomed.common.registry import registry
from pefomed.tasks.base_task import BaseTask
from pefomed.tasks.captioning import CaptionTask
from pefomed.tasks.image_text_pretrain import ImageTextPretrainTask
from pefomed.tasks.multimodal_classification import (
    MultimodalClassificationTask,
)
from pefomed.tasks.retrieval import RetrievalTask
from pefomed.tasks.vqa import VQATask, GQATask, AOKVQATask, DisCRNTask
from pefomed.tasks.vqa_reading_comprehension import VQARCTask, GQARCTask
from pefomed.tasks.dialogue import DialogueTask
from pefomed.tasks.text_to_image_generation import TextToImageGenerationTask
from pefomed.tasks.vg import MEDVGTask
from pefomed.tasks.vim_mamba import mambaVQATask

def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task

__all__ = [
    "BaseTask",
    "AOKVQATask",
    "RetrievalTask",
    "CaptionTask",
    "VQATask",
    "GQATask",
    "VQARCTask",
    "GQARCTask",
    "MultimodalClassificationTask",
    # "VideoQATask",
    # "VisualEntailmentTask",
    "ImageTextPretrainTask",
    "DialogueTask",
    "TextToImageGenerationTask",
    "DisCRNTask"
]
