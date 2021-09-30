import os
import numpy as np
import functions as f
import supervisely_lib as sly
from supervisely_lib.io.fs import mkdir

import interactive_demo.controller as ritm



project_id = 7246
image_id = 1655902

api = sly.Api.from_env()

project_info = api.project.get_info_by_id(project_id)
project_meta_json = api.project.get_meta(project_info.id)
project_meta = sly.ProjectMeta.from_json(project_meta_json)

work_dir = "/work/src/sly_dev/work_dir"
img_dir = os.path.join(work_dir, "img")
# ann_dir = os.path.join(work_dir, "ann")

mkdir(work_dir, True)
mkdir(img_dir)
# mkdir(ann_dir)

image_info = api.image.get_info_by_id(image_id)
api.image.download_path(image_id, os.path.join(img_dir, image_info.name))

img_path = os.path.join(img_dir, image_info.name)

ann_info = api.annotation.download(image_id)
ann_json = ann_info.annotation
ann = sly.Annotation.from_json(ann_json, project_meta)

clicks_map = f.get_points_from_image(ann.labels)
clicks_list = f.get_click_list_from_map(clicks_map)
image_np = api.image.download_np(image_id)

image = ritm.draw_with_blend_and_clicks(image_np, clicks_list=clicks_list)

from isegm.inference.predictors.base import BasePredictor
from isegm.inference.clicker import Clicker

path_to_model = "/work/src/models/coco_lvis_h32_itermask.pth"

# devices: cpu, cuda, xpu, mkldnn, opengl, opencl, ideep, hip, msnpu, mlc, xla, vulkan, meta, hpu
predictor = BasePredictor(path_to_model, "cpu")

clicker = Clicker()
for click in clicks_list:
    clicker.add_click(click)
print(clicker)

img = predictor.set_input_image(image_np)
pts = predictor.get_points_nd(clicks_list)
print(pts)


# results
from matplotlib import pyplot as plt
plt.imshow(image, interpolation='nearest')
plt.show()
