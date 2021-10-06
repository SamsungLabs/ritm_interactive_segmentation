import os
import timeit

import torch
import functools
import mask_image
import sly_globals as g
import sly_functions as f
import supervisely_lib as sly
from supervisely_lib.io.fs import silent_remove, mkdir

# diskcache==5.2.1

import time


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
    start_time = time.time()

    x1, y1, x2, y2 = f.get_smart_bbox(context["crop"])

    print("GET BBOX FROM CONTEXT--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    img_path = os.path.join(g.img_dir, "base_image.png")
    base_image_np = f.get_image_by_hash(context["image_hash"], img_path)

    print("DOWNLOAD IMAGE AND CACHE IT --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    bbox = sly.Rectangle(y1, x1, y2, x2)
    crop_np = sly.image.crop(base_image_np, bbox)

    pos_points, neg_points = f.get_pos_neg_points_list_from_context(context)
    pos_points, neg_points = f.get_pos_neg_points_list_from_context_bbox_relative(x1, y1, pos_points, neg_points)
    clicks_list = f.get_click_list_from_points(pos_points, neg_points)

    print("CROP AND GET CLICKS LIST --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    res_mask = mask_image.get_mask_from_clicks(crop_np, clicks_list)

    print("GET MASK FROM CLICKS --- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    bitmap = f.get_bitmap_from_mask(res_mask)
    bitmap_origin, bitmap_data = f.unpack_bitmap(bitmap, y1, x1)

    print("GET BITMAP --- %s seconds ---" % (time.time() - start_time))

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
