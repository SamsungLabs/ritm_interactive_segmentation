import os
import torch
import mask_image
import functions as f
import supervisely_lib as sly
from matplotlib import pyplot as plt
from supervisely_lib.io.fs import mkdir
from isegm.inference.utils import load_is_model


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

# image_info = api.image.get_info_by_id(image_id)
# api.image.download_path(image_id, os.path.join(img_dir, image_info.name))
#
# img_path = os.path.join(img_dir, image_info.name)

ann_info = api.annotation.download(image_id)
ann_json = ann_info.annotation
ann = sly.Annotation.from_json(ann_json, project_meta)

clicks_map = f.get_points_from_image(ann.labels)
clicks_list = f.get_click_list_from_map(clicks_map)
image_np = api.image.download_np(image_id)


path_to_model = "/work/models/coco_lvis_h32_itermask.pth"
device = "cpu"

model = torch.load(path_to_model, map_location=torch.device(device))
model = load_is_model(model, device)

res_mask = mask_image.get_mask_from_clicks(model, image_np, clicks_list)

plt.imshow(res_mask, interpolation='nearest')
#plt.imsave("/work/src/mask.png", res_mask)
plt.show()


