"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from pefomed.common.registry import registry
from pefomed.processors.base_processor import BaseProcessor
from pefomed.processors.blip_diffusion_processors import (
    BlipDiffusionInputImageProcessor,
    BlipDiffusionTargetImageProcessor,
)
from pefomed.processors.blip_processors import (
    BlipImageTrainProcessor,
    Blip2ImageTrainProcessor,
    BlipImageEvalProcessor,
    BlipCaptionProcessor,
)
from pefomed.processors.clip_processors import ClipImageTrainProcessor
from pefomed.processors.gpt_processors import (
    GPTVideoFeatureProcessor,
    GPTDialogueProcessor,
)
from pefomed.processors.instruction_text_processors import BlipInstructionProcessor
from pefomed.processors.ulip_processors import ULIPPCProcessor

__all__ = [
    "BaseProcessor",
    # BLIP
    "BlipImageTrainProcessor",
    "Blip2ImageTrainProcessor",
    "BlipImageEvalProcessor",
    "BlipCaptionProcessor",
    "BlipInstructionProcessor",
    # BLIP-Diffusion
    "BlipDiffusionInputImageProcessor",
    "BlipDiffusionTargetImageProcessor",
    # CLIP
    "ClipImageTrainProcessor",
    # GPT
    "GPTVideoFeatureProcessor",
    "GPTDialogueProcessor",
    # 3D
    "ULIPPCProcessor",
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
