import os
import sly_globals as g
from matplotlib import pyplot as plt
from interactive_demo.controller import InteractiveController
# brs_modes = ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']


def get_mask_from_clicks(model, image_np, clicks_list, device="cpu", brs_mode="NoBRS"):
    predictor_params = {'brs_mode': brs_mode}
    controller = InteractiveController(model, device, predictor_params)
    controller.set_image(image_np)
    for click in clicks_list:
        controller.add_click(click.coords[1], click.coords[0], click.is_positive)

    res_mask = controller.result_mask
    mask_path = os.path.join(g.mask_dir, "smart_mask.png")

    plt.imsave(mask_path, res_mask, cmap="Greys")
    # plt.imshow(res_mask, cmap="Greys", interpolation='nearest')
    # plt.show()
    #sly.image.write(mask_path, res_mask) dont work (black image)
    return res_mask
