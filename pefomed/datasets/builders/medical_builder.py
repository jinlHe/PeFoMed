from pefomed.common.registry import registry
from pefomed.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from pefomed.datasets.datasets.medical.clef_dataset import CLEFDataset
from pefomed.datasets.datasets.medical.medicat_dataset import MEDICATDataset
from pefomed.datasets.datasets.medical.report.iuxray_dataset import IUXRAYDataset,IUXRAYEvalDataset
from pefomed.datasets.datasets.medical.report.mimiccxr_dataset import MIMICCXRDataset, MIMICCXREvalDataset

from pefomed.datasets.datasets.medical.vqa_datasets import *
from pefomed.datasets.datasets.medical.roco_dataset import (
    ROCODataset,
    ROCOEvalDataset
)

@registry.register_builder("clef")
class CLEFBuilder(BaseDatasetBuilder):
    train_dataset_cls = CLEFDataset
    eval_dataset_cls = CLEFDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medical/clef/defaults.yaml",
    }


@registry.register_builder("mimiccxr")
class MIMICCXRBuilder(BaseDatasetBuilder):
    train_dataset_cls = MIMICCXRDataset
    eval_dataset_cls = MIMICCXREvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medical/mimiccxr/defaults.yaml",
    }


@registry.register_builder("iuxray")
class IUXRAYBuilder(BaseDatasetBuilder):
    train_dataset_cls = IUXRAYDataset
    eval_dataset_cls = IUXRAYEvalDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medical/iuxray/defaults.yaml",
    }


@registry.register_builder("roco")
class ROCOBuilder(BaseDatasetBuilder):
    train_dataset_cls = ROCODataset
    eval_dataset_cls = ROCOEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medical/roco/defaults.yaml",
    }


@registry.register_builder("medicat")
class MEDICATBuilder(BaseDatasetBuilder):
    train_dataset_cls = MEDICATDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medical/medicat/defaults.yaml",
    }


@registry.register_builder("slake")
class SLAKEBuilder(BaseDatasetBuilder):
    train_dataset_cls = MEDVQADataset
    eval_dataset_cls = MEDVQAEvalData

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medical/slake/defaults.yaml",
    }


@registry.register_builder("pathvqa")
class PATHVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = MEDVQADataset
    eval_dataset_cls = MEDVQAEvalData

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medical/pathvqa/defaults.yaml",
    }


@registry.register_builder("vqarad")
class VQARADBuilder(BaseDatasetBuilder):
    train_dataset_cls = MEDVQADataset
    eval_dataset_cls = MEDVQAEvalData

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/medical/vqarad/defaults.yaml",
    }
