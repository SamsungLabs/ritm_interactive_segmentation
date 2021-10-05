import torch
import mask_image
import sly_demo_functions as df
import supervisely_lib as sly
from matplotlib import pyplot as plt
from isegm.inference.utils import load_is_model


project_id = 7246
image_id = 1655902

api = sly.Api.from_env()

project_info = api.project.get_info_by_id(project_id)
project_meta_json = api.project.get_meta(project_info.id)
project_meta = sly.ProjectMeta.from_json(project_meta_json)


ann_info = api.annotation.download(image_id)
ann_json = ann_info.annotation
ann = sly.Annotation.from_json(ann_json, project_meta)

clicks_map = df.get_points_from_image(ann.labels)
clicks_list = df.get_click_list_from_map(clicks_map)
image_np = api.image.download_np(image_id)


path_to_model = "/work/ritm/models/coco_lvis_h32_itermask.pth"
device = "cpu"

model = torch.load(path_to_model, map_location=torch.device(device))
model = load_is_model(model, device)

res_mask = mask_image.get_mask_from_clicks(model, image_np, clicks_list)

plt.imshow(res_mask, interpolation='nearest')
plt.show()
