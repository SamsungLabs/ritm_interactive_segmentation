import os
import torch
import functools
import mask_image
import sly_globals as g
import sly_functions as f
import supervisely_lib as sly
from isegm.inference.utils import load_is_model
from supervisely_lib.io.fs import silent_remove, mkdir


def send_error_data(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            request_id = kwargs["context"]["request_id"]
            g.my_app.send_response(request_id, data={"error": repr(e)})
        return value

    return wrapper


@g.my_app.callback("smart_segmentation")
@sly.timeit
@send_error_data
def smart_segmentation(api: sly.Api, task_id, context, state, app_logger):
    img_path = os.path.join(g.img_dir, "base_image.png")
    base_image_np = f.get_image_by_hash(context["image_hash"], img_path)

    pos_points, neg_points = f.get_pos_neg_points_list_from_context(context)
    clicks_list = f.get_click_list_from_points(pos_points, neg_points)

    path_to_model = "/work/models/coco_lvis_h32_itermask.pth"
    device = "cpu"
    model = torch.load(path_to_model, map_location=torch.device(device))
    model = load_is_model(model, device)

    res_mask = mask_image.get_mask_from_clicks(model, base_image_np, clicks_list)
    # bitmap = f.get_bitmap_from_mask(res_mask)

    bitmap = f.get_bitmap_from_points(pos_points, neg_points)
    bitmap_origin, bitmap_data = f.unpack_bitmap(bitmap)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data={"origin": bitmap_origin, "bitmap": bitmap_data, "success": True, "error": None})


def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.TEAM_ID,
        "context.workspaceId": g.WORKSPACE_ID,
        "context.projectId": g.PROJECT_ID
    })

    g.my_app.run()


if __name__ == "__main__":
    sly.main_wrapper("main", main)
