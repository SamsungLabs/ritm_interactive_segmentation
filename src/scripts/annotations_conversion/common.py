from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
from tqdm import tqdm
import numpy as np


def parallel_map(array, worker, const_args=None, n_jobs=16, use_kwargs=False, front_num=3, drop_none=False):
    """
        A parallel version of the map function with a progress bar.

        Args:
            array (array-like): A list to iterate over
            worker (function): A python function to apply to the elements of array
            const_args (dict, default=None): Constant arguments, shared between all processes
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of
                keyword arguments to function
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job
            drop_none (boolean, default=False): Whether to drop None values from the list of results or not
        Returns:
            [worker(**list[0], **const_args), worker(**list[1], **const_args), ...]
    """
    # Replace None with empty dict
    const_args = dict() if const_args is None else const_args
    # We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [worker(**a, **const_args) if use_kwargs else worker(a, **const_args) for a in array[:front_num]]
    else:
        front = []
    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [worker(**a, **const_args) if use_kwargs else
                        worker(a, **const_args) for a in tqdm(array[front_num:])]
    # Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(worker, **a, **const_args) for a in array[front_num:]]
        else:
            futures = [pool.submit(worker, a, **const_args) for a in array[front_num:]]
        tqdm_kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True,
            'ncols': 100
        }
        # Print out the progress as tasks complete
        for _ in tqdm(as_completed(futures), **tqdm_kwargs):
            pass
    out = []
    # Get the results from the futures.
    for i, future in enumerate(futures):
        try:
            out.append(future.result())
        except Exception as e:
            print(f"Caught {str(e)} on {i}-th input.")
            out.append(None)

    if drop_none:
        return [v for v in front+out if v is not None]
    else:
        return front + out


def get_masks_hierarchy(masks, masks_meta):
    order = sorted(list(enumerate(masks_meta)), key=lambda x: x[1][1])[::-1]
    hierarchy = defaultdict(list)

    def check_inter(i, j):
        assert masks_meta[i][1] >= masks_meta[j][1]
        bbox_i, bbox_j = masks_meta[i][0], masks_meta[j][0]
        bbox_score = get_bbox_intersection(bbox_i, bbox_j) / get_bbox_area(bbox_j)
        if bbox_score < 0.7:
            return False

        mask_i, mask_j = masks[i], masks[j]
        mask_score = np.logical_and(mask_i, mask_j).sum() / masks_meta[j][1]
        return mask_score > 0.8

    def get_root_indx(root_indx, check_indx):
        children = hierarchy[root_indx]
        for child_indx in children:
            if masks_meta[child_indx][1] < masks_meta[check_indx][1]:
                continue
            result_indx = get_root_indx(child_indx, check_indx)
            if result_indx is not None:
                return result_indx

        if check_inter(root_indx, check_indx):
            return root_indx

        return None

    used_masks = np.zeros(len(masks), dtype=np.bool)
    parents = [None] * len(masks)
    node_level = [0] * len(masks)
    for ti in range(len(masks) - 1):
        for tj in range(ti + 1, len(masks)):
            i = order[ti][0]
            j = order[tj][0]

            assert i != j
            if used_masks[j] or not check_inter(i, j):
                continue

            ni = get_root_indx(i, j)
            assert ni != j and parents[j] is None
            hierarchy[ni].append(j)
            used_masks[j] = True
            parents[j] = ni
            node_level[j] = node_level[ni] + 1

    hierarchy = [hierarchy[i] for i in range(len(masks))]
    hierarchy = {i: {'children': hierarchy[i],
                     'parent': parents[i],
                     'node_level': node_level[i]
                     }
                 for i in range(len(masks))}
    return hierarchy


def get_bbox_intersection(b1, b2):
    h_i = get_segments_intersection(b1[:2], b2[:2])
    w_i = get_segments_intersection(b1[2:4], b2[2:4])
    return h_i * w_i


def get_segments_intersection(s1, s2):
    a, b = s1
    c, d = s2
    return max(0, min(b, d) - max(a, c) + 1)


def get_bbox_area(bbox):
    return (bbox[1] - bbox[0] + 1) * (bbox[3] - bbox[2] + 1)


def get_iou(mask1, mask2):
    intersection_area = np.logical_and(mask1, mask2).sum()
    union_area = np.logical_or(mask1, mask2).sum()
    return intersection_area / union_area


def encode_masks(masks):
    layers = [np.zeros(masks[0].shape, dtype=np.uint8)]
    layers_objs = [[]]
    objs_mapping = [(None, None)] * len(masks)
    ordered_masks = sorted(list(enumerate(masks)), key=lambda x: x[1].sum())[::-1]
    for global_id, obj_mask in ordered_masks:
        for layer_indx, (layer_mask, layer_objs) in enumerate(zip(layers, layers_objs)):
            if len(layer_objs) >= 255:
                continue
            if np.all(layer_mask[obj_mask] == 0):
                layer_objs.append(global_id)
                local_id = len(layer_objs)
                layer_mask[obj_mask] = local_id
                objs_mapping[global_id] = (layer_indx, local_id)
                break
        else:
            new_layer = np.zeros_like(layers[-1])
            new_layer[obj_mask] = 1
            objs_mapping[global_id] = (len(layers), 1)
            layers.append(new_layer)
            layers_objs.append([global_id])

    layers = [cv2.imencode('.png', x)[1] for x in layers]
    return layers, objs_mapping


def decode_masks(packed_data):
    layers, objs_mapping = packed_data
    layers = [cv2.imdecode(x, cv2.IMREAD_UNCHANGED) for x in layers]
    masks = []
    for layer_indx, obj_id in objs_mapping:
        masks.append(layers[layer_indx] == obj_id)
    return masks
