import os
import supervisely_lib as sly
from supervisely_lib.io.fs import mkdir

my_app = sly.AppService()
api: sly.Api = my_app.public_api

TEAM_ID = int(os.environ['context.teamId'])
WORKSPACE_ID = int(os.environ['context.workspaceId'])
PROJECT_ID = int(os.environ["modal.state.slyProjectId"])

DEVICE = "cpu"

work_dir = "/work/ritm/src/sly_dev/work_dir"
img_dir = os.path.join(work_dir, "img")
mask_dir = os.path.join(work_dir, "mask")

model_path = "/work/ritm/models/coco_lvis_h32_itermask.pth"

mkdir(work_dir, True)
mkdir(img_dir)
mkdir(mask_dir)


# results
# from matplotlib import pyplot as plt
# plt.imshow(image, interpolation='nearest')
# plt.show()
