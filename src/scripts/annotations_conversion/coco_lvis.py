import cv2
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

from isegm.data.datasets import LvisDataset, CocoDataset
from isegm.utils.misc import get_bbox_from_mask, get_bbox_iou
from scripts.annotations_conversion.common import get_masks_hierarchy, get_iou, encode_masks


def create_annotations(lvis_path: Path, coco_path: Path, dataset_split='train', min_object_area=80):
    lvis_dataset = LvisDataset(lvis_path, split=dataset_split)
    lvis_samples = lvis_dataset.dataset_samples
    lvis_annotations = lvis_dataset.annotations

    coco_dataset = CocoDataset(coco_path, split=dataset_split + '2017')

    coco_lvis_mapping = []
    lvis_images = {x['coco_url'].split('/')[-1].split('.')[0]: lvis_indx
                   for lvis_indx, x in enumerate(lvis_samples)}
    for indx, coco_sample in enumerate(coco_dataset.dataset_samples):
        lvis_indx = lvis_images.get(coco_sample['file_name'].split('.')[0], None)
        if lvis_indx is not None:
            coco_lvis_mapping.append((indx, lvis_indx))

    output_masks_path = lvis_path / dataset_split / 'masks'
    output_masks_path.mkdir(parents=True, exist_ok=True)

    hlvis_annotation = dict()
    for coco_indx, lvis_indx in tqdm(coco_lvis_mapping):
        coco_sample = get_coco_sample(coco_dataset, coco_indx)

        lvis_info = lvis_samples[lvis_indx]
        lvis_annotation = lvis_annotations[lvis_info['id']]
        empty_mask = np.zeros((lvis_info['height'], lvis_info['width']))
        image_name = lvis_info['coco_url'].split('/')[-1].split('.')[0]

        lvis_masks = []
        lvis_bboxes = []
        for obj_annotation in lvis_annotation:
            obj_mask = lvis_dataset.get_mask_from_polygon(obj_annotation, empty_mask)
            obj_mask = obj_mask == 1
            if obj_mask.sum() >= min_object_area:
                lvis_masks.append(obj_mask)
                lvis_bboxes.append(get_bbox_from_mask(obj_mask))

        coco_bboxes = []
        coco_masks = []
        for inst_id in coco_sample['instances_info'].keys():
            obj_mask = coco_sample['instances_mask'] == inst_id
            if obj_mask.sum() >= min_object_area:
                coco_masks.append(obj_mask)
                coco_bboxes.append(get_bbox_from_mask(obj_mask))

        masks = []
        for coco_j, coco_bbox in enumerate(coco_bboxes):
            for lvis_i, lvis_bbox in enumerate(lvis_bboxes):
                if get_bbox_iou(lvis_bbox, coco_bbox) > 0.70 and \
                        get_iou(lvis_masks[lvis_i], coco_masks[coco_j]) > 0.70:
                    break
            else:
                masks.append(coco_masks[coco_j])

        for ti, (lvis_mask, lvis_bbox) in enumerate(zip(lvis_masks, lvis_bboxes)):
            for tj_mask, tj_bbox in zip(lvis_masks[ti + 1:], lvis_bboxes[ti + 1:]):
                bbox_iou = get_bbox_iou(lvis_bbox, tj_bbox)
                if bbox_iou > 0.7 and get_iou(lvis_mask, tj_mask) > 0.85:
                    break
            else:
                masks.append(lvis_mask)

        masks_meta = [(get_bbox_from_mask(x), x.sum()) for x in masks]
        if not masks:
            continue

        hierarchy = get_masks_hierarchy(masks, masks_meta)

        for obj_id, obj_info in list(hierarchy.items()):
            if obj_info['parent'] is None and len(obj_info['children']) == 0:
                hierarchy[obj_id] = None

        merged_mask = np.max(masks, axis=0)
        num_instance_masks = len(masks)
        for obj_id in coco_sample['semantic_info'].keys():
            obj_mask = coco_sample['semantic_map'] == obj_id
            obj_mask = np.logical_and(obj_mask, np.logical_not(merged_mask))
            if obj_mask.sum() > 500:
                masks.append(obj_mask)

        hlvis_annotation[image_name] = {
            'num_instance_masks': num_instance_masks,
            'hierarchy': hierarchy
        }

        with open(output_masks_path / f'{image_name}.pickle', 'wb') as f:
            pickle.dump(encode_masks(masks), f)

    with open(lvis_path / dataset_split / 'hannotation.pickle', 'wb') as f:
        pickle.dump(hlvis_annotation, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_coco_sample(dataset, index):
    dataset_sample = dataset.dataset_samples[index]

    image_path = dataset.images_path / dataset.get_image_name(dataset_sample['file_name'])
    label_path = dataset.labels_path / dataset_sample['file_name']

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED).astype(np.int32)
    label = 256 * 256 * label[:, :, 0] + 256 * label[:, :, 1] + label[:, :, 2]

    instance_map = np.full_like(label, 0)
    semantic_map = np.full_like(label, 0)
    semantic_info = dict()
    instances_info = dict()
    for segment in dataset_sample['segments_info']:
        class_id = segment['category_id']
        obj_id = segment['id']
        if class_id not in dataset._things_labels_set:
            semantic_map[label == obj_id] = obj_id
            semantic_info[obj_id] = {'ignore': False}
            continue

        instance_map[label == obj_id] = obj_id
        ignore = segment['iscrowd'] == 1
        instances_info[obj_id] = {
            'ignore': ignore
        }

    sample = {
        'image': image,
        'instances_mask': instance_map,
        'instances_info': instances_info,
        'semantic_map': semantic_map,
        'semantic_info': semantic_info
    }

    return sample
