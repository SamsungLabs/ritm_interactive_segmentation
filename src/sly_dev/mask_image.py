from interactive_demo.controller import InteractiveController
# brs_modes = ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']


def get_mask_from_clicks(model, image_np, clicks_list, device="cpu", brs_mode="NoBRS"):
    predictor_params = {'brs_mode': brs_mode}
    controller = InteractiveController(model, device, predictor_params)
    controller.set_image(image_np)
    for click in clicks_list:
        controller.add_click(click.coords[1], click.coords[0], click.is_positive)

    res_mask = controller.result_mask
    return res_mask
