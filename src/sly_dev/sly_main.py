import os
import torch
import functools
import mask_image
import sly_globals as g
import sly_functions as f
import supervisely_lib as sly
from isegm.inference.utils import load_is_model
from supervisely_lib.io.fs import silent_remove, mkdir

# diskcache==5.2.1


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

#'nGfUNH5yJ9m19X4zb6/OGmIvGGL21wnr9f0K1/FQuLY='

@g.my_app.callback("smart_segmentation")
@sly.timeit
@send_error_data
def smart_segmentation(api: sly.Api, task_id, context, state, app_logger):
    img_path = os.path.join(g.img_dir, "base_image.png")
    base_image_np = f.get_image_by_hash(context["image_hash"], img_path)

    pos_points, neg_points = f.get_pos_neg_points_list_from_context(context)
    clicks_list = f.get_click_list_from_points(pos_points, neg_points)

    model = torch.load(g.model_path, map_location=torch.device(g.DEVICE))
    model = load_is_model(model, g.DEVICE)

    res_mask = mask_image.get_mask_from_clicks(model, base_image_np, clicks_list)
    bitmap = f.get_bitmap_from_mask(res_mask)

    bitmap_origin, bitmap_data = f.unpack_bitmap(bitmap)

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data={"origin": bitmap_origin, "bitmap": bitmap_data, "success": True, "error": None})


#@TODO: crop image before processing
def main():
    sly.logger.info("Script arguments", extra={
        "context.teamId": g.TEAM_ID,
        "context.workspaceId": g.WORKSPACE_ID,
        "context.projectId": g.PROJECT_ID
    })

    g.my_app.run()


if __name__ == "__main__":
    sly.main_wrapper("main", main)