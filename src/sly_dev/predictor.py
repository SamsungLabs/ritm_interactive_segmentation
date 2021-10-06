from isegm.inference.predictors.base import BasePredictor
from isegm.inference.clicker import Clicker


def get_prediction_from_clicks(model, image_np, clicks_list, device="cpu"):
    predictor = BasePredictor(model, device)
    predictor.set_input_image(image_np)
    clicker = Clicker()
    for click in clicks_list:
        clicker.add_click(click)
    prediction_image = predictor.get_prediction(clicker)
    return prediction_image
