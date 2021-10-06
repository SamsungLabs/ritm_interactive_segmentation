import os
import supervisely_lib as sly
from supervisely_lib.io.fs import mkdir
from diskcache import Cache


my_app = sly.AppService()
api: sly.Api = my_app.public_api

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
PROJECT_ID = int(os.environ["modal.state.slyProjectId"])


work_dir = "/work/ritm/src/sly_dev/work_dir"
mkdir(work_dir, True)

img_dir = os.path.join(work_dir, "img")

model_path = "/work/ritm/models/coco_lvis_h32_itermask.pth"
cache_dir = "/work/ritm/src/sly_dev/work_dir/img_cache"
cache = Cache(directory=cache_dir)
cache_item_limit = 30
mkdir(cache_dir)
# mkdir(img_dir)

# DEVICE
# devices: cpu, cuda, xpu, mkldnn, opengl, opencl,
# ideep, hip, msnpu, mlc, xla, vulkan, meta, hpu

DEVICE = "cpu"

# LOAD MODEL
import torch
from isegm.inference.utils import load_is_model

model = torch.load(model_path, map_location=torch.device(DEVICE))
model = load_is_model(model, DEVICE)

# RITM CONTROLLER
from interactive_demo.controller import InteractiveController
# brs_modes = ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']

brs_mode = "NoBRS"
predictor_params = {'brs_mode': brs_mode}
controller = InteractiveController(model, DEVICE, predictor_params)
