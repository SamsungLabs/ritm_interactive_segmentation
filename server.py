from operator import pos
# import flask
import matplotlib.pyplot as plt
import sys, random, os, json, glob
import numpy as np
import torch, cv2
from isegm.utils import vis, exp
from isegm.inference import utils
from isegm.inference.evaluation import evaluate_dataset, evaluate_sample
from isegm.inference.predictors import get_predictor
from isegm.inference.clicker import Clicker, Click
from skimage import measure
from PIL import Image, ImageDraw, ImageFont
from imantics import Polygons, Mask, Annotation
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from service_streamer import ThreadedStreamer, Streamer
import urllib.request
import ssl
from flask_cors import CORS, cross_origin

EVAL_MAX_CLICKS = 20
MODEL_THRESH = 0.49
TARGET_IOU = 0.95
TEMP_PATH = 'temp/'
os.makedirs(TEMP_PATH, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg = exp.load_config_file('./config.yml', return_edict=True)
checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, 'coco_lvis_h18_itermask')
model = utils.load_is_model(checkpoint_path, device)

# Possible choices: 'NoBRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C', 'RGB-BRS', 'DistMap-BRS'
brs_mode = 'f-BRS-B'
predictor = get_predictor(model, brs_mode, device, prob_thresh=MODEL_THRESH)


# start service
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 10 # 10MB max
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
cors = CORS(app)

@app.route("/interactive_segmentation", methods=["POST"])
def main():
    """Main method for interactive segmentation
    """
    img_file = request.files.get('image', None)
    gt_mask_file = request.files.get('gt_mask', None)
    click_history = eval(request.form['click_history'])
    prev_polygon = eval(request.form['prev_polygon'])
    file_url:str = request.form.get('file_url', None)
    prev_mask = request.files.get('prev_mask', None) 
    tolerance = int(request.form.get('tolerance', 1))
    view_img = request.form.get('view_img', False)
    view_img = False
    # view_img = True if view_img.lower() == 'true' else False
    img:Image = None
    filename:str = None

    # Read file from cache or from OSS or from http request
    if img_file:
        filename = secure_filename(img_file.filename)
        img = Image.open(img_file)
    else:
        # try load from temp first
        filename = file_url.split('/')[-1]
        file_path = TEMP_PATH+filename
        if os.path.exists(file_path):
            img = Image.open(file_path)
        elif file_url.startswith('oss://'): #TODO
            # if region is the same:
                # try to load file from OSS internal
                # img = ...
            # else:
                # try to load file from OSS external
                # img = ...
            img.save(file_path)
        elif file_url: #TODO
            # filename = file_url.split('/')[-1]
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib.request.urlretrieve(file_url, file_path)
            # load file from url
            img = Image.open(file_path)
            # img.save(file_path)
    
    # processing imputs
    img_np, clicks, prev_mask = processing_inputs(img, click_history, prev_polygon)
    
    # make prediction
    pred_probs = streamer.predict([(img_np, clicks, prev_mask)])
    
    # prepare result
    results = prepare_result(img_np, pred_probs, clicks, gt_mask_file, tolerance, view_img, filename)

    # return
    return jsonify(results)

def processing_inputs(img:Image, click_history:list, prev_polygon:list):
    """processing inputs for model

    Args:
        img (Image): pillow Image format

        click_history (list): a list of clicks dict as [{x:x, y:y, positive:p}, ...]

        prev_polygon (list): polygon shaped as (n, m, 2): [[(x, y), (x, y)...], [...], ...], where n is enclosed polygons, m is the length of coordinates
    """
    img_np = np.asarray(img, dtype=np.uint8)
    #gen click history
    clicks = Clicker()
    for c in click_history:
        x = c['x']
        y = c['y']
        positive = c['positive']
        click = Click(is_positive=positive, coords=(y,x))
        clicks.add_click(click)

    #gen prev mask
    polygon = Polygons(prev_polygon)
    mask = polygon.mask(width=img.width, height=img.height).array
    prev_mask = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    return img_np, clicks, prev_mask
        

def predict(batch):
    #TODO: increase batch size
    assert len(batch) == 1, 'Only one sample is allowed at a time'
    img = batch[0][0].copy()
    clicks = batch[0][1]
    prev_mask = batch[0][2]
    with torch.no_grad():
        predictor.set_input_image(img)
        pred_probs = predictor.get_prediction(clicks, prev_mask)
    return [pred_probs] # resume wrapped


def prepare_result(img:np.ndarray, pred_probs:np.ndarray, clicks:Clicker, gt_mask_file, tolerance:int=1, view_img:bool=False, filename:str=None):
    """prepare result

    Args:
        img (np.ndarray): img in numpy.ndarray 
        pred_mask (np.ndarray): predicted mask from model
        clicks(Clicker): Cliker object for click history
        gt_mask_file (FileStorage): ground truth mask file
        tolerance (int, optional): Precision to convert from mask to polygon in pixel. Defaults to 1.
        view_img (bool, optional): Return result image url. Defaults to False.
    """
    # gen mask
    assert len(pred_probs) == 1, f'Only one output is expected, but got {len(pred_probs)}'
    pred_probs = pred_probs[0]
    pred_mask = pred_probs > MODEL_THRESH

    # convert mask to polygon
    regions = Mask(pred_mask).polygons().points
    polygons = []
    for polygon in regions:
        polygon2 = measure.approximate_polygon(polygon, tolerance)
        polygons.append(polygon2.tolist())
    results = {'polygons': polygons}

    # calculate iou
    if gt_mask_file:
        gt_mask = Image.open(gt_mask_file)
        mask_np = np.asarray(gt_mask, dtype=np.int32)
        if len(mask_np.shape) > 2:
            assert len(mask_np.shape) == 3
            mask_np = np.max(mask_np, axis=2)
        mask_np = np.where(mask_np>0, 1, 0)
        iou = utils.get_iou(mask_np, pred_mask)
        results['iou'] = iou
        print(iou)

    # save img with minor delay
    if view_img:
        ext = filename.split('.')[-1]
        draw = vis.draw_with_blend_and_clicks(img, mask=pred_mask, clicks_list=clicks.clicks_list)
        filename = filename.split('.')[0] + f'[{len(clicks.clicks_list)}].jpg'
        result_path = TEMP_PATH + filename
        Image.fromarray(draw).save(result_path)
        # return send_file(result_path)
        results['result'] = filename

    return results

@app.route("/view_image/<filename>", methods=["GET"])
def view_image(filename:str):
    path = TEMP_PATH+filename
    if not os.path.exists(path):
        return 'Image not found!'
    return send_file(path)
        

if __name__ == "__main__":
    streamer = ThreadedStreamer(predict, batch_size=1, max_latency=0.01)
    app.run(port=5005, debug=True, host= '0.0.0.0')
    print('Flask started')