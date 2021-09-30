import os
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
