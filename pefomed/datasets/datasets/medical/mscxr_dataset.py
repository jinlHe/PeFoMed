import os
import json
import pickle
import random
import time
import itertools

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pefomed.datasets.datasets.base_dataset import BaseDataset


class MSCXRDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.refer = REFER(ann_paths, vis_root)
        self.ref_ids = self.refer.getRefIds()

        self.instruction_pool = [
            "[refer] give me the location of {}",
            "[refer] where is {} ?",
            "[refer] from this image, tell me the location of {}",
            "[refer] the location of {} is",
            "[refer] could you tell me the location for {} ?",
            "[refer] where can I locate the {} ?",
        ]

    def preprocess(self, index):
        ref_id = self.ref_ids[index]
        ref = self.refer.loadRefs(ref_id)[0]

        image_file = self.refer.getImgpath(ref['image_id'])[0]
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image_orig_size = image.size
        image = self.vis_processor(image)
        image_new_size = [image.shape[1], image.shape[2]]

        image_new_size = [100, 100]

        sample_sentence = ref['label_text']
        refer_sentence = self.text_processor(sample_sentence)

        bbox = self.refer.getRefBox(ref['id'])
        bbox = [
            bbox[0] / image_orig_size[0] * image_new_size[0],
            bbox[1] / image_orig_size[1] * image_new_size[1],
            (bbox[0] + bbox[2]) / image_orig_size[0] * image_new_size[0],
            (bbox[1] + bbox[3]) / image_orig_size[1] * image_new_size[1]
        ]
        bbox = [int(x) for x in bbox]
        bbox = "{{<{}><{}><{}><{}>}}".format(*bbox)
        return {
            "image": image,
            "ann_id": ref['id'],
            "refer_sentence": refer_sentence,
            "bbox": bbox,
            "image_id": ref['image_id'],
        }

    def __len__(self):
        return len(self.annotation['annotations'])

    def __getitem__(self, index):
        data = self.preprocess(index)
        instruction = random.choice(self.instruction_pool).format(data['refer_sentence'])

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        instruction = self.text_processor(instruction)

        return {
            "image": data['image'],
            "ann_id":data['ann_id'],
            "text_input": instruction,
            "answer": data['bbox'],
            "image_id": data['image_id'],
        }


class MSCXREvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.refer = REFER(ann_paths, vis_root)
        self.ref_ids = self.refer.getRefIds()

    def preprocess(self, index):
        ref_id = self.ref_ids[index]
        ref = self.refer.loadRefs(ref_id)[0]

        image_file = self.refer.getImgpath(ref['image_id'])[0]
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image_orig_size = image.size
        image = self.vis_processor(image)
        image_new_size = [image.shape[1], image.shape[2]]

        image_new_size = [100,100]

        sample_sentence = ref['label_text']
        refer_sentence = self.text_processor(sample_sentence)

        bbox = self.refer.getRefBox(ref['id'])
        bbox = [
            bbox[0] / image_orig_size[0] * image_new_size[0],
            bbox[1] / image_orig_size[1] * image_new_size[1],
            (bbox[0] + bbox[2]) / image_orig_size[0] * image_new_size[0],
            (bbox[1] + bbox[3]) / image_orig_size[1] * image_new_size[1]
        ]
        bbox = [int(x) for x in bbox]
        bbox = "{{<{}><{}><{}><{}>}}".format(*bbox)
        return {
            "image": image,
            "ann_id": ref['id'],
            "refer_sentence": refer_sentence,
            "bbox": bbox,
            "image_id": ref['image_id'],
        }

    def __len__(self):
        return len(self.annotation['annotations'])

    def __getitem__(self, index):
        data = self.preprocess(index)
        instruction = "<Img><ImageHere></Img> [refer] give me the location of {}".format(data['refer_sentence'])
        instruction = self.text_processor(instruction)

        return {
            "image": data['image'],
            "ann_id": data['ann_id'],
            "text_input": instruction,
            "answers": data['bbox'],
            "image_id": data['image_id'],
        }


class REFER:
    def __init__(self, data_root, vis_root):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        print('loading dataset mscxr into memory...')
        # print("REFER data_root: {}".format(data_root)) #['/mnt/sda/hjl/data/medical/ms_cxr/MS_CXR_Local_Alignment_v1.0.0.json']
        # print("vis_root: {}".format(vis_root)) #/mnt/sdb/lpf/data/physionet.org/files/mimic-cxr-jpg/2.0.0
        self.ann_dir = data_root[0]
        self.vis_root = vis_root

        # load refs from data/dataset/refs(dataset).json
        tic = time.time()
        self.data = {}
        # self.data['dataset'] = dataset
        # self.data['refs'] = json.load(open(ref_file, 'rb'))
        with open(self.ann_dir, 'r') as f: self.data['refs'] = json.load(f)

        # load annotations from data/dataset/instances.json

        self.data['images'] = self.data['refs']['images']
        self.data['annotations'] = self.data['refs']['annotations']
        self.data['categories'] = self.data['refs']['categories']

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns,imgToImgs = {}, {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
            imgToImgs[img['id']] = imgToImgs.get(img['id'], []) + [img]
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['annotations']:
            # ids
            ref_id = ref['id']
            ann_id = ref['id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToImgs = imgToImgs
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        print('index created.')

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data['annotations']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if
                            split[-1] in ref['split']]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    raise 'No such split [%s]' % split
        ref_ids = [ref['id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def getImgpath(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.imgToImgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.imgToImgs[image_ids][0]['path']]

    def loadCats(self, cat_ids=[]):
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ref = self.Refs[ref_id]
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]

    def showRef(self, ref, seg_box='box'):
        ax = plt.gca()
        # show image
        image = self.Imgs[ref['image_id']]
        I = io.imread(os.path.join(self.vis_root, image['file_name']))
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('%s. %s' % (sid + 1, sent['sent']))
        # show segmentations
        if seg_box == 'seg':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = 'none'
            if type(ann['segmentation'][0]) == list:
                # polygon used for refcoco*
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg) / 2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for refclef
                raise NotImplementedError('RefClef is not downloaded')
        # show bounding-box
        elif seg_box == 'box':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref['ref_id'])
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(box_plot)
