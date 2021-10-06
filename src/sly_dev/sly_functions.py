import cv2
import numpy as np
import sly_globals as g
import supervisely_lib as sly
from isegm.inference.clicker import Click
from supervisely_lib.io.fs import silent_remove


def get_smart_bbox(crop):
    x1, y1 = crop[0]["x"], crop[0]["y"]
    x2, y2 = crop[1]["x"], crop[1]["y"]
    return x1, y1, x2, y2


def get_click_list_from_points(pos_points, neg_points):
    clicks_list = []
    for coords in pos_points:
        click = Click(True, (coords[1], coords[0]))
        clicks_list.append(click)

    for coords in neg_points:
        click = Click(False, (coords[1], coords[0]))
        clicks_list.append(click)
    return clicks_list


def get_pos_neg_points_list_from_context(context):
    pos_points = context["positive"]
    neg_points = context["negative"]

    pos_points_list = []
    for coords in pos_points:
        pos_point = []
        for coord in coords:
            pos_point.append(coords[coord])
        pos_points_list.append(pos_point)

    neg_points_list = []
    for coords in neg_points:
        neg_point = []
        for coord in coords:
            neg_point.append(coords[coord])
        neg_points_list.append(neg_point)
    return pos_points_list, neg_points_list


def get_bitmap_from_points(pos_points, neg_points):
    mask = np.zeros((800, 1067, 3), np.uint8)
    for pos_point in pos_points:
        cv2.circle(mask, (pos_point[0], pos_point[1]), 15, (255, 255, 255), -1)
    for neg_point in neg_points:
        cv2.circle(mask, (neg_point[0], neg_point[1]), 15, (0, 0, 0), -1)
    sly.image.write(f'{g.my_app.data_dir}/sly_base_sir/images/smart_mask.png', mask)
    mask = mask[..., 0]
    bool_mask = np.array(mask, dtype=bool)
    bitmap = sly.Bitmap(bool_mask)
    return bitmap


def unpack_bitmap(bitmap):
    bitmap_json = bitmap.to_json()["bitmap"]
    bitmap_origin = bitmap_json["origin"]
    bitmap_origin = {"y": bitmap_origin[1], "x": bitmap_origin[0]}

    bitmap_data = bitmap_json["data"]
    return bitmap_origin, bitmap_data


def get_image_by_hash(hash, save_path):
    if g.cache.get(hash) is None:
        g.api.image.download_paths_by_hashes([hash], [save_path])
        base_image = sly.image.read(save_path)
        g.cache.add(hash, base_image)
        silent_remove(save_path)
    else:
        base_image = g.cache.get(hash)
    if g.cache.count > g.cache_item_limit:
        g.cache.clear()
    return base_image


def get_bitmap_from_mask(mask):
    bool_mask = np.array(mask, dtype=bool)
    bitmap = sly.Bitmap(bool_mask)
    return bitmap
