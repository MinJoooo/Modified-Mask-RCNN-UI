import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from io import StringIO

# Root directory of the project and import MaskRCNN
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class BalloonConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "balloon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + objects

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # Add classes. (Imported from utils.py)
        self.add_class("balloon", 1, "heart")
        self.add_class("balloon", 2, "star")
        print("class_info:", self.class_info)


        # Train or validation dataset?
        assert subset in ["train", "val"] ### If subset is not 'train' or 'val', call AssertionError
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json"))) ### Required "import json"
        annotations = list(annotations.values()) # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]


        # Add images
        for a in annotations:
            regions = a['regions'].values()

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "balloon",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                regions=regions)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count]
            with one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id] ### 'image_info' is created by method 'utils.add_image'
        if image_info["source"] != "balloon":
            return super(self.__class__, self).load_mask(image_id)


        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["regions"])],
                        dtype=np.uint8)
        class_ids = []

        for i, p in enumerate(info["regions"]):
            # Get indexes of pixels inside the polygon and set them to 1
            p_points = p["shape_attributes"]
            rr, cc = skimage.draw.polygon(p_points['all_points_y'], p_points['all_points_x'])
            mask[rr, cc, i] = 1

            p_class = p["region_attributes"]["display_name"]
            for info in self.class_info:
                if info["name"] == p_class:
                    class_ids.append(info["id"])
        class_ids = np.array(class_ids)

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), class_ids


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def training(image_path, model_path, config):

    # Configurations and create model
    config = BalloonConfig()
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=DEFAULT_LOGS_DIR)

    # Load weights
    if model_path is not None:
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)

    # Training dataset.
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(image_path, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BalloonDataset()
    dataset_val.load_balloon(image_path, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # No need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=3,
                layers='heads')