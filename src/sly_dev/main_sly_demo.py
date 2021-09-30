import os
import functions as f
import supervisely_lib as sly
from supervisely_lib.io.fs import mkdir


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

clicks = f.get_points_from_image(ann.labels)
print(clicks)

