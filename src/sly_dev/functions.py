import os
import supervisely_lib as sly


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

