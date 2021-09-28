import torch
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from isegm.data.datasets import *
from isegm.model.losses import *
from isegm.data.transforms import *
from isegm.engine.trainer import ISTrainer
from isegm.model.metrics import AdaptiveIoU
from isegm.data.points_sampler import MultiPointSampler
from isegm.utils.log import logger
from isegm.model import initializer

from isegm.model.is_hrnet_model import HRNetModel
from isegm.model.is_deeplab_model import DeeplabModel