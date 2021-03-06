import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
from isegm.utils import vis, exp
from isegm.inference import utils
from isegm.inference.evaluation import evaluate_dataset, evaluate_sample
from isegm.inference.predictors import get_predictor
from isegm.inference.clicker import Clicker
EVAL_MAX_CLICKS = 20
MODEL_THRESH = 0.49
TARGET_IOU = 0.95

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg = exp.load_config_file('./config.yml', return_edict=True)
checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, 'coco_lvis_h18_itermask')
model = utils.load_is_model(checkpoint_path, device)

# Possible choices: 'NoBRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C', 'RGB-BRS', 'DistMap-BRS'
brs_mode = 'f-BRS-B'
predictor = get_predictor(model, brs_mode, device, prob_thresh=MODEL_THRESH)

# test from dataset
DATASET = 'GrabCut'
dataset = utils.get_dataset(DATASET, cfg)
sample_id = 12
sample = dataset.get_sample(sample_id)
gt_mask = sample.gt_mask

# clicks_list, ious_arr, pred = evaluate_sample(sample.image, gt_mask, predictor, 
#                                               pred_thr=MODEL_THRESH, 
#                                               max_iou_thr=TARGET_IOU, max_clicks=EVAL_MAX_CLICKS)
clicker = Clicker(gt_mask=gt_mask)
pred_mask = np.zeros_like(gt_mask)
ious_list = []

