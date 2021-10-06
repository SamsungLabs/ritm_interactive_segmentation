import cv2
import numpy as np
import supervisely_lib as sly
from isegm.inference.clicker import Click


def get_points_from_image(labels):
    clicks = {"pos": [], "neg": []}
    for label in labels:
        if label.geometry.geometry_name() == "point":
            if label.obj_class.name == "pos":
                click_coords = label.geometry.to_json()["points"]["exterior"][0]
                clicks["pos"].append(click_coords)
            if label.obj_class.name == "neg":
                click_coords = label.geometry.to_json()["points"]["exterior"][0]
                clicks["neg"].append(click_coords)
    return clicks


def get_click_list_from_map(clicks_map):
    clicks_list = []
    for coords in clicks_map["pos"]:
        click = Click(True, (coords[1], coords[0]))
        clicks_list.append(click)

    for coords in clicks_map["neg"]:
        click = Click(False, (coords[1], coords[0]))
        clicks_list.append(click)
    return clicks_list


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