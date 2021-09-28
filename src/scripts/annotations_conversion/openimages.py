import csv
import pickle as pkl
from pathlib import Path
from collections import defaultdict


def create_annotations(dataset_path, dataset_split='train'):
    dataset_path = Path(dataset_path)
    _split_path = dataset_path / dataset_split
    _images_path = _split_path / 'images'
    _masks_path = _split_path / 'masks'
    clean_anno_path = _split_path / f'{dataset_split}-annotations-object-segmentation_clean.pkl'

    annotations = {
        'image_id_to_masks': defaultdict(list),  # mapping from image_id to a list of masks
        'dataset_samples': []  # list of unique image ids
    }

    with open(_split_path / f'{dataset_split}-annotations-object-segmentation.csv', 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            image_id = row['ImageID']
            mask_path = row['MaskPath']

            if (_images_path / f'{image_id}.jpg').is_file() \
                    and (_masks_path / mask_path).is_file():
                annotations['image_id_to_masks'][image_id].append(mask_path)
    annotations['dataset_samples'] = list(annotations['image_id_to_masks'].keys())

    with clean_anno_path.open('wb') as f:
        pkl.dump(annotations, f)

    return annotations
