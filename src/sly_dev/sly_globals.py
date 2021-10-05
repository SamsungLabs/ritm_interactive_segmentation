import os
import supervisely_lib as sly
from supervisely_lib.io.fs import mkdir
from diskcache import Cache


my_app = sly.AppService()
api: sly.Api = my_app.public_api

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
PROJECT_ID = int(os.environ["modal.state.slyProjectId"])

DEVICE = "cpu"
work_dir = "/work/ritm/src/sly_dev/work_dir"
mkdir(work_dir, True)

img_dir = os.path.join(work_dir, "img")

model_path = "/work/ritm/models/coco_lvis_h32_itermask.pth"
cache_dir = "/work/ritm/src/sly_dev/work_dir/img_cache"
cache = Cache(directory=cache_dir)
cache_item_limit = 30
mkdir(cache_dir)
# mkdir(img_dir)
