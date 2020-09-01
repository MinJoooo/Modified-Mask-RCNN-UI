import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project and import MaskRCNN, COCO
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
import mrcnn.model as modellib
import samples.balloon2.balloon2_new as balloon

# Directory to save logs and trained model
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class DemoConfig(balloon.BalloonConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    DETECTION_MIN_CONFIDENCE = 0.85


############################################################
#  Dataset
############################################################

def load_class_names():
    class_names = ['BG', 'heart', 'star']
    return class_names


############################################################
#  Main function
############################################################

def training(image_path, model_path):

    # Configurations and create model
    config = DemoConfig()
    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=DEFAULT_LOGS_DIR)

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Load class_names
    class_names = load_class_names()

    # Load image file
    image = skimage.io.imread(image_path)

    # Run detection
    results, text_image, text_molded_images, text_image_metas, text_anchors = model.detect([image], verbose=1)

    # Visualize results
    r = results[0]
    filename = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
    path = os.getcwd()
    path = os.path.join(path, filename)

    return path, text_image, text_molded_images, text_image_metas, text_anchors


### It doesn't run if model is not proper or low accuracy.