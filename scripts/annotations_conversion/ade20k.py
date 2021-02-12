import pickle as pkl
from pathlib import Path
from scipy.io import loadmat

from scripts.annotations_conversion.common import parallel_map


ADE20K_STUFF_CLASSES = ['water', 'wall', 'snow', 'sky', 'sea', 'sand', 'road', 'route', 'river', 'path', 'mountain',
                        'mount', 'land', 'ground', 'soil', 'hill', 'grass', 'floor', 'flooring', 'field', 'earth',
                        'ground', 'fence', 'ceiling', 'wave', 'crosswalk', 'hay bale', 'bridge', 'span', 'building',
                        'edifice', 'cabinet', 'cushion', 'curtain', 'drape', 'drapery', 'mantle', 'pall', 'door',
                        'fencing', 'house', 'pole', 'seat', 'windowpane', 'window', 'tree', 'towel', 'table',
                        'stairs', 'steps', 'streetlight', 'street lamp', 'sofa', 'couch', 'lounge', 'skyscraper',
                        'signboard', 'sign', 'sidewalk', 'pavement', 'shrub', 'bush', 'rug', 'carpet']


def worker_annotations_loader(anno_pair, dataset_path):
    image_id, folder = anno_pair
    n_masks = len(list((dataset_path / folder).glob(f'{image_id}_*.png')))

    # each image has several layers with instances,
    # each layer has mask name and instance_to_class mapping
    layers = [{
        'mask_name': f'{image_id}_{suffix}.png',
        'instance_to_class': {},
        'object_instances': [],
        'stuff_instances': []
    } for suffix in ['seg'] + [f'parts_{i}' for i in range(1, n_masks)]]

    # parse txt with instance to class mappings
    with (dataset_path / folder / (image_id + "_atr.txt")).open('r') as f:
        for line in f:
            # instance_id layer_n is_occluded class_names class_name_raw attributes
            line = line.strip().split('#')
            inst_id, layer_n, class_names = int(line[0]), int(line[1]), line[3]

            # there may be more than one class name for each instance
            class_names = [name.strip() for name in class_names.split(',')]

            # check if any of classes is stuff
            if set(class_names) & set(ADE20K_STUFF_CLASSES):
                layers[layer_n]['stuff_instances'].append(inst_id)
            else:
                layers[layer_n]['object_instances'].append(inst_id)
            layers[layer_n]['instance_to_class'][inst_id] = class_names

    return layers


def load_and_parse_annotations(dataset_path, dataset_split, n_jobs=1):
    dataset_split_folder = 'training' if dataset_split == 'train' else 'validation'

    orig_annotations = loadmat(dataset_path / 'index_ade20k.mat', squeeze_me=True, struct_as_record=True)
    image_ids = [image_id.split('.')[0] for image_id in orig_annotations['index'].item()[0]
                 if dataset_split in image_id]
    folders = [Path(folder).relative_to('ADE20K_2016_07_26') for folder in orig_annotations['index'].item()[1]
               if dataset_split_folder in folder]

    # list of dictionaries with filename and instance to class mapping
    all_layers = parallel_map(list(zip(image_ids, folders)), worker_annotations_loader, n_jobs=n_jobs,
                              use_kwargs=False, const_args={
                                'dataset_path': dataset_path
                              })

    return image_ids, folders, all_layers


def create_annotations(dataset_path, dataset_split='train', n_jobs=1):
    anno_path = dataset_path / f'{dataset_split}-annotations-object-segmentation.pkl'
    image_ids, folders, all_layers = load_and_parse_annotations(dataset_path, dataset_split, n_jobs=n_jobs)

    # create dictionary with annotations
    annotations = {}
    for index, image_id in enumerate(image_ids):
        annotations[image_id] = {
            'folder': folders[index],
            'layers': all_layers[index]
        }

    with anno_path.open('wb') as f:
        pkl.dump(annotations, f)

    return annotations
