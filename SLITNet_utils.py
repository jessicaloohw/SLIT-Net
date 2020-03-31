# SLIT-Net
# DOI: 10.1109/JBHI.2020.2983549

import os
import multiprocessing
import h5py as hh
import numpy as np
import csv
import skimage.io
import skimage.transform
import skimage.segmentation
import skimage.measure
from skimage.measure import find_contours
from scipy.ndimage.morphology import binary_fill_holes
from scipy.spatial.distance import directed_hausdorff

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot
from matplotlib import patches
from matplotlib.patches import Polygon

import mrcnn.utils as utils

import SLITNet_model as modellib

######################################### CONFIGURATION #############################################

class SLITNetConfig(object):

    def __init__(self):

        # Note: Do not change this name because the SLIT-Net workflow depends on it.
        self.NAME = "ulcer"

        # Number of GPUs to use. When using only a CPU, this needs to be set to 1.
        self.GPU_COUNT = 1

        # Number of images to train with on each GPU
        self.IMAGES_PER_GPU = 5

        # Number of training steps per epoch
        self.STEPS_PER_EPOCH = 19

        # Number of validation steps to run at the end of every training epoch
        self.VALIDATION_STEPS = 4

        # Backbone network architecture
        # Supported values are: resnet50, resnet101.
        # You can also provide a callable that should have the signature
        # of model.resnet_graph. If you do so, you need to supply a callable
        # to COMPUTE_BACKBONE_SHAPE as well
        self.BACKBONE = "resnet50"

        # Only useful if you supply a callable to BACKBONE.
        # Should compute the shape of each layer of the FPN.
        # See model.compute_backbone_shapes()
        self.COMPUTE_BACKBONE_SHAPE = None

        # The strides of each layer of the FPN.
        # These values are based on a Resnet50/Resnet101 backbone.
        self.BACKBONE_STRIDES = [4, 8, 16, 32, 64]

        # Size of the fully-connected layers in the classification graph
        self.FPN_CLASSIF_FC_LAYERS_SIZE = 1024

        # Size of the top-down layers used to build the FPN
        self.TOP_DOWN_PYRAMID_SIZE = 256

        # Number of classification classes (including background)
        # Note: 8 for diffuse white light images
        # Note: 5 for diffuse blue light images
        self.NUM_CLASSES = 5

        # Length of square anchor side in pixels
        self.RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

        # Ratios of anchors at each cell (width/height)
        self.RPN_ANCHOR_RATIOS = [0.5, 1, 2]

        # Anchor stride
        self.RPN_ANCHOR_STRIDE = 1

        # Non-max suppression threshold to filter RPN proposals
        self.RPN_NMS_THRESHOLD = 0.7

        # How many anchors per image to use for RPN training
        self.RPN_TRAIN_ANCHORS_PER_IMAGE = 256

        # ROIs kept after tf.nn.top_k and before non-maximum suppression
        self.PRE_NMS_LIMIT = 6000

        # ROIs kept after non-maximum suppression (training and inference)
        self.POST_NMS_ROIS_TRAINING = 2000
        self.POST_NMS_ROIS_INFERENCE = 1000

        # If enabled, resizes instance masks to a smaller size to reduce memory load
        self.USE_MINI_MASK = False
        self.MINI_MASK_SHAPE = (56, 56)

        # Input image resizing
        # Generally, use the "square" resizing mode for training and predicting
        # and it should work well in most cases. In this mode, images are scaled
        # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
        # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
        # padded with zeros to make it a square so multiple images can be put
        # in one batch.
        # Available resizing modes:
        # none:   No resizing or padding. Return the image unchanged.
        # square: Resize and pad with zeros to get a square image
        #         of size [max_dim, max_dim].
        # pad64:  Pads width and height with zeros to make them multiples of 64.
        #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
        #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
        #         The multiple of 64 is needed to ensure smooth scaling of feature
        #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
        # crop:   Picks random crops from the image. First, scales the image based
        #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
        #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
        #         IMAGE_MAX_DIM is not used in this mode.
        self.IMAGE_RESIZE_MODE = "square"
        self.IMAGE_MIN_DIM = 512
        self.IMAGE_MAX_DIM = 512
        # Minimum scaling ratio. Checked after MIN_IMAGE_DIM and can force further
        # up scaling. For example, if set to 2 then images are scaled up to double
        # the width and height, or more, even if MIN_IMAGE_DIM doesn't require it.
        # Howver, in 'square' mode, it can be overruled by IMAGE_MAX_DIM.
        self.IMAGE_MIN_SCALE = 0
        # Number of color channels per image. RGB = 3, grayscale = 1, RGB-D = 4
        # Changing this requires other changes in the code. See the WIKI for more
        # details: https://github.com/matterport/Mask_RCNN/wiki
        self.IMAGE_CHANNEL_COUNT = 3

        # Image mean (RGB)
        # Note: [103.7, 68.2, 49.6] for diffuse white light images
        # Note: [12.7, 45.4, 69.4] for diffuse blue light images
        # self.MEAN_PIXEL = np.array([103.7, 68.2, 49.6])
        self.MEAN_PIXEL = np.array([12.7, 45.4, 69.4])

        # Number of ROIs per image to feed to classifier/mask heads
        self.TRAIN_ROIS_PER_IMAGE = 100

        # Percent of positive ROIs used to train classifier/mask heads
        self.ROI_POSITIVE_RATIO = 0.33

        # Pooled ROIs
        self.POOL_SIZE = 7

        # Shape of output mask
        self.MASK_POOL_SIZE = 14
        self.MASK_SHAPE = [28, 28]

        # Parameters for loss functions
        self.MASK_HAUSDORFF_POWER = 2.0
        self.RPN_CLASS_FOCAL_POWER = 2.0
        self.MRCNN_CLASS_FOCAL_POWER = 2.0

        # Maximum number of ground truth instances to use in one image
        self.MAX_GT_INSTANCES = 100

        # Bounding box refinement standard deviation for RPN and final detections.
        self.RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
        self.BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

        # Max number of final detections
        self.DETECTION_MAX_INSTANCES = 100

        # Minimum probability value to accept a detected instance
        self.DETECTION_MIN_CONFIDENCE = 0.5

        # Non-maximum suppression threshold for detection
        self.DETECTION_NMS_THRESHOLD = 0.5

        # Learning rate and momentum
        self.OPTIMIZER_PARAMS = {"lr": 0.001,
                                 "momentum": 0.9,
                                 "decay": 1e-06,
                                 "nesterov": True,
                                 "clipnorm": 5.0
                                 }

        # Weight decay regularization
        self.WEIGHT_DECAY = 0.0001

        # Loss weights
        self.LOSS_WEIGHTS = {
            "rpn_class_loss": 1.0,
            "rpn_bbox_loss": 1.0,
            "mrcnn_class_loss": 1.0,
            "mrcnn_bbox_loss": 1.0,
            "mrcnn_mask_loss": 1.0
        }

        # Use RPN ROIs or externally generated ROIs for training
        self.USE_RPN_ROIS = True

        # Train or freeze batch normalization layers
        #     None: Train BN layers. This is the normal mode. Will set layer to false during inference.
        #     False: Freeze BN layers. Good when using a small batch size
        #     True: (don't use). Set layer in training mode even when predicting
        self.TRAIN_BN = None

        # Multi-processing
        # Generates data on multiple threads, but may cause data duplication
        self.USE_MULTIPROCESSING = False
        if(self.USE_MULTIPROCESSING):
            self.NUM_WORKERS = multiprocessing.cpu_count()
        else:
            self.NUM_WORKERS = 1

        ##############################################################################################
        #                         DERIVED ATTRIBUTES (DO NOT CHANGE)                                 #
        ##############################################################################################

        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                                         self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                                         self.IMAGE_CHANNEL_COUNT])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES


    def set_to_inference_mode(self, class_threshold=0.5, nms_threshold=0.5):

        print("Setting config to inference mode...")

        # In inference, we evaluate one image at a time:
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 1
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Set thresholds:
        self.DETECTION_MIN_CONFIDENCE = class_threshold
        self.DETECTION_NMS_THRESHOLD = nms_threshold


    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


    def write_to_file(self, filename):
        """ Write Configuration values to file. """
        with open(filename, 'w') as f:
            f.write("Configurations:\n")
            for a in dir(self):
                if not a.startswith("__") and not callable(getattr(self, a)):
                    f.write("{:30} {} \n".format(a, getattr(self, a)))



######################################### DATA HANDLERS #############################################

class SLITNetDataset(utils.Dataset):

    def load_image(self, image_id):

        image = skimage.io.imread(self.image_info[image_id]["path"])

        return image


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """

        # Masks:
        mask = self.image_info[image_id]["masks"]
        mask = np.transpose(np.array(mask), (1,2,0))
        mask = mask.astype('bool')

        # Class IDs:
        class_ids = self.image_info[image_id]["labels"]
        class_ids = np.reshape(np.array(class_ids), len(class_ids))
        class_ids = class_ids.astype('int')

        return mask, class_ids


    def image_reference(self, image_id):

        return self.image_info[image_id]["path"]


    def get_ground_truth_in_original_size(self, image_id):

        labels = self.image_info[image_id]["labels"]
        boxes = self.image_info[image_id]["boxes"] # (x, y, w, h)
        masks = self.image_info[image_id]["masks"]

        # Initialise:
        numObjects = len(labels)
        ground_truth = {"class_ids": np.zeros(numObjects, np.int32),
                        "rois": np.zeros([numObjects, 4], np.int32), # (y1, x1, y2, x2)
                        "masks": np.zeros([masks[0].shape[0], masks[0].shape[1], numObjects], np.bool)
                        }

        for i in range(numObjects):

            # Label:
            try:
                class_id = labels[i][0, 0]
            except:
                class_id = labels[i]

            # Bounding box:
            x1, y1, w, h = boxes[i][:, 0]
            x2 = x1 + w
            y2 = y1 + h

            # Mask:
            mask = masks[i].astype(bool)

            # Update dictionary:
            ground_truth["class_ids"][i] = class_id
            ground_truth["rois"][i, :] = [y1, x1, y2, x2]
            ground_truth["masks"][:, :, i] = mask

        return ground_truth


class SLITNetDataset_white(SLITNetDataset):

    def load_dataset(self, filepath):

        # Get source name:
        source = filepath.split('/')
        source = source[len(source) - 1]
        source = source.split('.')
        source = source[0]

        # Open file:
        f = hh.File(filepath)

        # Get data:
        data = f['data']

        # Get number of images:
        nImages = data['IMAGE'].shape[0]

        # Read data:
        for i in range(nImages):

            # Path to image:
            path = ''
            path_ref = data['PATH'][i,0]
            path_to_decode = data[path_ref].value
            for p in path_to_decode:
                path = path + chr(p)

            # References:
            labels_ref = data['LABEL'][i,0]
            boxes_ref = data['BOUNDING_BOX'][i,0] # [x y w h]
            masks_ref = data['MASK'][i,0]

            # For each object:
            nObjects = data[labels_ref].shape[0]

            labels = []
            boxes = []
            masks = []

            for j in range(nObjects):

                label_ref = data[labels_ref].value[j,0]
                box_ref = data[boxes_ref].value[j,0]
                mask_ref = data[masks_ref].value[j,0]

                label = data[label_ref].value
                box = data[box_ref].value
                mask = data[mask_ref].value
                mask = np.transpose(mask, (1,0))

                labels.append(label)
                boxes.append(box)
                masks.append(mask)

            # Add image:
            self.add_image(source=source,
                           image_id=i,
                           path=path,
                           labels=labels,
                           boxes=boxes,
                           masks=masks)

        # Add classes:
        self.add_class(source=source, class_id=1, class_name="ulcer")
        self.add_class(source=source, class_id=2, class_name="white_cells")
        self.add_class(source=source, class_id=3, class_name="hypopyon")
        self.add_class(source=source, class_id=4, class_name="edema")
        self.add_class(source=source, class_id=5, class_name="reflex")
        self.add_class(source=source, class_id=6, class_name="pupil")
        self.add_class(source=source, class_id=7, class_name="limbus")


class SLITNetDataset_blue(SLITNetDataset):

    def load_dataset(self, filepath):

        # Get source name:
        source = filepath.split('/')
        source = source[len(source) - 1]
        source = source.split('.')
        source = source[0]

        # Open file:
        f = hh.File(filepath)

        # Get data:
        data = f['data']

        # Get number of images:
        nImages = data['IMAGE_BLUE'].shape[0]

        # Read data:
        for i in range(nImages):

            # Path to image:
            path = ''
            path_ref = data['PATH_BLUE'][i,0]
            path_to_decode = data[path_ref].value
            for p in path_to_decode:
                path = path + chr(p)

            # References:
            labels_ref = data['LABEL_BLUE'][i,0]
            boxes_ref = data['BOUNDING_BOX_BLUE'][i,0] # [x y w h]
            masks_ref = data['MASK_BLUE'][i,0]

            # For each object:
            nObjects = data[labels_ref].shape[0]

            labels = []
            boxes = []
            masks = []

            for j in range(nObjects):

                label_ref = data[labels_ref].value[j,0]
                box_ref = data[boxes_ref].value[j,0]
                mask_ref = data[masks_ref].value[j,0]

                label = data[label_ref].value
                box = data[box_ref].value
                mask = data[mask_ref].value
                mask = np.transpose(mask, (1,0))

                labels.append(label)
                boxes.append(box)
                masks.append(mask)

            # Add image:
            self.add_image(source=source,
                           image_id=i,
                           path=path,
                           labels=labels,
                           boxes=boxes,
                           masks=masks)

        # Add classes:
        self.add_class(source=source, class_id=1, class_name="epidefect")
        self.add_class(source=source, class_id=2, class_name="reflex")
        self.add_class(source=source, class_id=3, class_name="pupil")
        self.add_class(source=source, class_id=4, class_name="limbus")



############################################ HELPER FUNCTIONS ########################################

def plot_learning_curves(model_dir, sub_dir, min_epoch_val=0):

    # Get file:
    filename = os.path.join(model_dir, sub_dir, 'loss.csv')

    # Read data from file:
    epochs = []

    train_loss = []
    train_mrcnn_bbox = []
    train_mrcnn_class = []
    train_mrcnn_mask = []
    train_rpn_bbox = []
    train_rpn_class = []

    val_loss = []
    val_mrcnn_bbox = []
    val_mrcnn_class = []
    val_mrcnn_mask = []
    val_rpn_bbox = []
    val_rpn_class = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:

            epochs.append(int(row['epoch']))

            train_loss.append(float(row['loss']))
            train_mrcnn_bbox.append(float(row['mrcnn_bbox_loss']))
            train_mrcnn_class.append(float(row['mrcnn_class_loss']))
            train_mrcnn_mask.append(float(row['mrcnn_mask_loss']))
            train_rpn_bbox.append(float(row['rpn_bbox_loss']))
            train_rpn_class.append(float(row['rpn_class_loss']))

            val_loss.append(float(row['val_loss']))
            val_mrcnn_bbox.append(float(row['val_mrcnn_bbox_loss']))
            val_mrcnn_class.append(float(row['val_mrcnn_class_loss']))
            val_mrcnn_mask.append(float(row['val_mrcnn_mask_loss']))
            val_rpn_bbox.append(float(row['val_rpn_bbox_loss']))
            val_rpn_class.append(float(row['val_rpn_class_loss']))

    epochs = np.array(epochs)

    train_loss = np.array(train_loss)
    train_mrcnn_bbox = np.array(train_mrcnn_bbox)
    train_mrcnn_class = np.array(train_mrcnn_class)
    train_mrcnn_mask = np.array(train_mrcnn_mask)
    train_rpn_bbox = np.array(train_rpn_bbox)
    train_rpn_class = np.array(train_rpn_class)

    val_loss = np.array(val_loss)
    val_mrcnn_bbox = np.array(val_mrcnn_bbox)
    val_mrcnn_class = np.array(val_mrcnn_class)
    val_mrcnn_mask = np.array(val_mrcnn_mask)
    val_rpn_bbox = np.array(val_rpn_bbox)
    val_rpn_class = np.array(val_rpn_class)

    # Plot:
    plot_dir = os.path.join(model_dir, sub_dir, 'learning_curves_min_epoch_val={}'.format(min_epoch_val))
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # Get index of min_epoch_val:
    min_idx = np.where(epochs == min_epoch_val)[0][0]

    pyplot.figure()
    pyplot.plot(epochs, train_loss, 'b', label='training')
    pyplot.plot(epochs, val_loss, 'r', label='validation')
    pyplot.legend()
    pyplot.title('loss | min @ epoch {} | min_epoch_val = {}'.format(epochs[np.argmin(val_loss[min_idx:]) + min_idx], min_epoch_val))
    pyplot.savefig(os.path.join(plot_dir, 'loss.png'))
    pyplot.close()

    pyplot.figure()
    pyplot.plot(epochs, train_mrcnn_bbox, 'b', label='training')
    pyplot.plot(epochs, val_mrcnn_bbox, 'r', label='validation')
    pyplot.legend()
    pyplot.title('mrcnn bbox loss | min @ epoch {} | min_epoch_val = {}'.format(epochs[np.argmin(val_mrcnn_bbox[min_idx:]) + min_idx], min_epoch_val))
    pyplot.savefig(os.path.join(plot_dir, 'mrcnn_bbox_loss.png'))
    pyplot.close()

    pyplot.figure()
    pyplot.plot(epochs, train_mrcnn_class, 'b', label='training')
    pyplot.plot(epochs, val_mrcnn_class, 'r', label='validation')
    pyplot.legend()
    pyplot.title('mrcnn class loss | min @ epoch {} | min_epoch_val = {}'.format(epochs[np.argmin(val_mrcnn_class[min_idx:]) + min_idx], min_epoch_val))
    pyplot.savefig(os.path.join(plot_dir, 'mrcnn_class_loss.png'))
    pyplot.close()

    pyplot.figure()
    pyplot.plot(epochs, train_mrcnn_mask, 'b', label='training')
    pyplot.plot(epochs, val_mrcnn_mask, 'r', label='validation')
    pyplot.legend()
    pyplot.title('mrcnn mask loss | min @ epoch {} | min_epoch_val = {}'.format(epochs[np.argmin(val_mrcnn_mask[min_idx:]) + min_idx], min_epoch_val))
    pyplot.savefig(os.path.join(plot_dir, 'mrcnn_mask_loss.png'))
    pyplot.close()

    pyplot.figure()
    pyplot.plot(epochs, train_rpn_bbox, 'b', label='training')
    pyplot.plot(epochs, val_rpn_bbox, 'r', label='validation')
    pyplot.legend()
    pyplot.title('rpn bbox loss | min @ epoch {} | min_epoch_val = {}'.format(epochs[np.argmin(val_rpn_bbox[min_idx:]) + min_idx], min_epoch_val))
    pyplot.savefig(os.path.join(plot_dir, 'rpn_bbox_loss.png'))
    pyplot.close()

    pyplot.figure()
    pyplot.plot(epochs, train_rpn_class, 'b', label='training')
    pyplot.plot(epochs, val_rpn_class, 'r', label='validation')
    pyplot.legend()
    pyplot.title('rpn class loss | min @ epoch {} | min_epoch_val = {}'.format(epochs[np.argmin(val_rpn_class[min_idx:]) + min_idx], min_epoch_val))
    pyplot.savefig(os.path.join(plot_dir, 'rpn_class_loss.png'))
    pyplot.close()


def restrict_within_limbus(results, IDS_TO_RESTRICT=[1,2,3,4,6], LIMBUS_ID=7, HIGH_OVERLAP=0.7):

    # Initialise:
    results_clean = {"class_ids": results["class_ids"],
                     "rois": results["rois"],
                     "masks": results["masks"],
                     "scores": results["scores"]
                     }
    toDelete = np.zeros([results_clean["class_ids"].shape[0], 1])  # If 1, delete index

    # Only one limbus can exist:
    limbus_idxs = np.where(results_clean["class_ids"] == LIMBUS_ID)[0]
    if(len(limbus_idxs) == 0):
        print('No limbus detected.')
        return results_clean

    elif (len(limbus_idxs) > 1):
        print('Only one limbus can exist.')
        limbus_scores = results_clean["scores"][limbus_idxs]
        best_limbus_idx = limbus_idxs[np.argmax(limbus_scores)]
        for idx in limbus_idxs:
            if not (idx == best_limbus_idx):
                toDelete[idx] = 1

    else:
        best_limbus_idx = limbus_idxs[0]

    # Get limbus mask:
    limbus_mask = results_clean["masks"][:, :, best_limbus_idx]

    # Update:
    results_clean = update_results(results_clean, toDelete)

    # Remove any ulcers, white cells, hypopyon, edema, pupil
    # that does not have a certain percentage within the limbus area:
    nInstances = results_clean["class_ids"].shape[0]
    for idx in range(nInstances):
        label = results_clean["class_ids"][idx]
        if(label in IDS_TO_RESTRICT):
            mask = results_clean["masks"][:, :, idx]
            mask_within_limbus = np.multiply(limbus_mask, mask)
            percentage_within_limbus = np.sum(mask_within_limbus)/np.sum(mask)
            if(percentage_within_limbus < HIGH_OVERLAP):
                toDelete[idx] = 1

    # Update:
    results_clean = update_results(results_clean, toDelete)

    return results_clean


def get_limbus_idx(idxs, LIMBUS_ID=7):
    limbus_idxs = np.where(idxs == LIMBUS_ID)[0]
    if(len(limbus_idxs) == 0):
        print('No limbus available.')
        return None
    elif(len(limbus_idxs) > 1):
        print('More than one limbus available. Please call restrict_within_limbus() first.')
        return None
    else:
        return limbus_idxs[0]


def update_results(results, to_delete):

    to_delete = np.where(to_delete)[0]
    results["class_ids"] = np.delete(results["class_ids"], to_delete, axis=0)
    results["rois"] = np.delete(results["rois"], to_delete, axis=0)
    results["masks"] = np.delete(results["masks"], to_delete, axis=2)
    results["scores"] = np.delete(results["scores"], to_delete, axis=0)

    return results


def calculate_dice_overlap(pred_mask, truth_mask):

    intersect_mask = np.multiply(pred_mask, truth_mask)
    pred_area = np.sum(pred_mask)
    truth_area = np.sum(truth_mask)
    intersection = np.sum(intersect_mask)
    union = pred_area + truth_area

    if (union == 0):
        dice = 1.0
    else:
        dice = (2 * intersection) / union
    return dice


def calculate_performance_metrics(pred, truth, num_classes):
    """
    Calculates per-class Dice overlap

    pred        = output of model.detect() after post-processing etc.
    truth       = ground truth
    num_classes = total number of classes, excluding background i.e. config.NUM_CLASSES - 1
    """

    # Ensure the dimensions are the same:
    if not (pred["masks"].shape[0:2] == truth["masks"].shape[0:2]):
        print('Performance statistics cannot be calculated because dimensions do not match.')
        return

    # Classes:
    classes = range(1, num_classes+1)
    class_dice = np.zeros([num_classes, 1])
    class_pred_area = np.zeros([num_classes, 1])
    class_truth_area = np.zeros([num_classes, 1])

    # Ensure masks are boolean:
    pred["masks"] = pred["masks"].astype(bool)
    truth["masks"] = truth["masks"].astype(bool)

    for n, c in enumerate(classes):

        # Initialise:
        truth_mask = np.zeros_like(truth["masks"][:, :, 0])
        pred_mask = np.zeros_like(truth_mask)

        # Get indices:
        pred_idxs = np.where(pred["class_ids"] == c)[0]
        truth_idxs = np.where(truth["class_ids"] == c)[0]

        # Get binary maps:
        for i in pred_idxs:
            pred_mask += pred["masks"][:, :, i]
        pred_mask = pred_mask.astype(bool)
        pred_area = np.sum(pred_mask.astype(int))

        for i in truth_idxs:
            truth_mask += truth["masks"][:, :, i]
        truth_mask = truth_mask.astype(bool)
        truth_area = np.sum(truth_mask.astype(int))

        # Calculate Dice:
        dice = calculate_dice_overlap(pred_mask, truth_mask)

        # Update:
        class_dice[n] = dice
        class_pred_area[n] = pred_area
        class_truth_area[n] = truth_area

    return class_dice, class_pred_area, class_truth_area


def calculate_hausdorff_metric(pred, truth, num_classes):

    # Ensure the dimensions are the same:
    if not (pred["masks"].shape[0:2] == truth["masks"].shape[0:2]):
        print('Performance statistics cannot be calculated because dimensions do not match.')
        return

    # Classes:
    classes = range(1, num_classes + 1)
    class_hausdorff = np.zeros([num_classes, 1])

    # Ensure masks are boolean:
    pred["masks"] = pred["masks"].astype(bool)
    truth["masks"] = truth["masks"].astype(bool)

    for n, c in enumerate(classes):

        # Initialise:
        truth_mask = np.zeros_like(truth["masks"][:, :, 0])
        pred_mask = np.zeros_like(truth_mask)

        # Get indices:
        pred_idxs = np.where(pred["class_ids"] == c)[0]
        truth_idxs = np.where(truth["class_ids"] == c)[0]

        # Get binary maps:
        for i in pred_idxs:
            pred_mask += pred["masks"][:, :, i]
        pred_mask = pred_mask.astype(bool)

        for i in truth_idxs:
            truth_mask += truth["masks"][:, :, i]
        truth_mask = truth_mask.astype(bool)

        # Get contours to calculate HD:
        truth_contours = skimage.measure.find_contours(truth_mask, 0.5)
        pred_contours = skimage.measure.find_contours(pred_mask, 0.5)
        nTruth = len(truth_contours)
        nPred = len(pred_contours)

        # If there's only one contour for each then it's fine, just calculate:
        if nTruth == 1 and nPred == 1:
            tc = truth_contours[0]
            pc = pred_contours[0]
            hd = max(directed_hausdorff(tc, pc)[0], directed_hausdorff(pc, tc)[0])

        # But if there are multiple instances of each class, find the best match,
        # and those with no matches will not contribute
        # e.g. if there is an extra prediction, it does not contribute
        # Average the HD for all matches for that class
        elif nTruth > 0 and nPred > 0:
            sum_hd = 0
            sum_cnt = 0
            invalid_idxs = []
            for it in range(nTruth):
                # Compare each truth to all preds:
                tc = truth_contours[it]
                hd_all = np.zeros([nPred])
                for ip in range(nPred):
                    # Check if this pred has been matched with something else:
                    if ip in invalid_idxs:
                        hd_all[ip] = np.inf
                    else:
                        pc = pred_contours[ip]
                        hd_all[ip] = max(directed_hausdorff(tc, pc)[0], directed_hausdorff(pc, tc)[0])
                    # Find the best match:
                    best_idx = np.argmin(hd_all)
                    best_val = hd_all[best_idx]
                    # Add to sum:
                    if not best_val == np.inf:
                        sum_hd += best_val
                        sum_cnt += 1
                        # Remove this index from future comparisons:
                        invalid_idxs.append(best_idx)
            # Average per class:
            hd = sum_hd / sum_cnt

        elif nTruth == 0 and nPred == 0:
            hd = 0

        else:
            hd = np.nan

        # Update:
        class_hausdorff[n] = hd

    return class_hausdorff


def get_predictions_in_small_size(results, threshold=0.5):
    """
    results     = output of model.detect()
    threshold   = to be applied to probability maps
    """

    # Initialise:
    numObjects = results["class_ids"].shape[0]
    results_small = {"class_ids": results["class_ids"],
                     "rois": results["rois"],
                     "masks": np.zeros_like(results["masks"]),
                     "scores": results["scores"]
                    }

    for i in range(numObjects):

        # Threshold:
        mask = results["masks"][:, :, i]
        mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

        # Update dictionary:
        results_small["masks"][:, :, i] = mask

    return results_small


def get_predictions_in_original_size(image_meta, results, threshold=0.5):
    """
    image_meta  = see compose_image_meta()
    results     = output of model.detect()
    threshold   = to be applied to probability maps
    """

    # Reshape image_meta if necessary:
    if(len(image_meta.shape) == 1):
        image_meta = np.reshape(image_meta, [1, image_meta.shape[0]])

    # Get meta-data:
    image_meta = modellib.parse_image_meta(image_meta)
    y_shift, x_shift, y_end, x_end =  image_meta["window"][0]
    h_orig, w_orig, _ = image_meta["original_image_shape"][0]
    h = y_end - y_shift
    w = x_end - x_shift
    h_scale = float(h_orig)/h
    w_scale = float(w_orig)/w

    # Initialise:
    numObjects = results["class_ids"].shape[0]
    results_orig = {"class_ids": np.zeros(numObjects, np.int32),
                    "rois": np.zeros([numObjects, 4], np.int32),
                    "masks": np.zeros([h_orig, w_orig, numObjects], np.bool),
                    "scores": np.zeros(numObjects, np.float32)
                    }

    # Scale to original size:
    for i in range(numObjects):

        # Bounding box:
        y1, x1, y2, x2 = results["rois"][i]
        y1 = int(h_scale*(y1 - y_shift))
        x1 = int(w_scale*(x1 - x_shift))
        y2 = int(h_scale*(y2 - y_shift))
        x2 = int(w_scale*(x2 - x_shift))

        # Masks:
        # Resize probability maps
        mask = results["masks"][y_shift:y_end, x_shift:x_end, i]
        mask = skimage.transform.resize(
            mask, (h_orig, w_orig),
            order=1, mode="constant", preserve_range=True)

        # Threshold:
        mask = np.where(mask >= threshold, 1, 0).astype(np.bool)

        # Update dictionary:
        results_orig["class_ids"][i] = results["class_ids"][i]
        results_orig["rois"][i, :] = [y1, x1, y2, x2]
        results_orig["masks"][:, :, i] = mask
        results_orig["scores"][i] = results["scores"][i]

    return results_orig


def get_predictions_of_interest(results, ids_of_interest=[1,2,3,4]):
    """
    results = ground truth or output of model.detect()
    ids_of_interest = [1, 2, 3, 4] where 1 = ulcer
                                         2 = white cells
                                         3 = hypopyon
                                         4 = edema

    Note: Assumes masks have already been thresholded by call to either
          get_prediction_in_small_size() or get_prediction_in_original_size()
    """

    # Find objects of interest:
    ids = np.where(np.isin(results["class_ids"], ids_of_interest))[0]

    # Initialise:
    numObjects = ids.shape[0]
    h, w, _ = results["masks"].shape
    results_interest = {"class_ids": np.zeros(numObjects, np.int32),
                        "rois": np.zeros([numObjects, 4], np.int32),
                        "masks": np.zeros([h, w, numObjects], np.bool)
                        }
    # Note: Ground truth doesn't have "scores"
    if 'scores' in results.keys():
        results_interest["scores"] = np.zeros(numObjects, np.float32)

    for n, i in enumerate(ids):

        # Update dictionary:
        results_interest["class_ids"][n] = results["class_ids"][i]
        results_interest["rois"][n, :] = results["rois"][i, :]
        results_interest["masks"][:, :, n] = results["masks"][:, :, i]
        if 'scores' in results.keys():
            results_interest["scores"][n] = results["scores"][i]

    return results_interest


def clean_predictions_white(results, rules=[]):
    """
    Cleans predictions by asserting some rules:

    MASKS:

    1. Segmentation masks have to be one piece,
       i.e. holes filled, small components removed.

    LABELS:

    2. If multiple of the same biomarker have HIGH overlap,
       only the one with the highest probability is retained.

    3. White cells cannot be smaller than ulcers,
       and white cells mask should include all the ulcer mask as well.

    4. If a reflex has a high probability,
       overlapping ulcers are removed.

    5. If a reflex has a low probability, and overlaps ulcers or white cells,
       the reflex is relabelled as an ulcer.

    6. White cells cannot exist alone (must overlap ulcers),
       if it has high probability, the white cells are relabelled as an ulcer,
       otherwise, it is deleted.

    7. Only one pupil can exist,
       only the one with the highest probability is retained, regardless of overlap.

    8. Only one hypopyon can exist,
       only the one with the highest probability is retained, regardless of overlap.

    9. Combine overlapping instances with the same label

        Choose only one:
    10. Edema cannot exist alone (must overlap ulcers or white cells),
        and edema mask should include all the ulcer and/or white cells masks as well.

    results = output of model.detect()
    rules   = which rules to assert; if empty, all are asserted

    Notes: Assumes masks have already been thresholded by call to either
           get_prediction_in_small_size() or get_prediction_in_original_size()
    """

    # Constants:
    NUM_RULES = 10
    HIGH_PROB = 0.8     # Threshold for which something is considered "high probability"
    HIGH_OVERLAP = 0.7  # Threshold for which something is considered to have "high overlap"

    # Check rules:
    if(len(rules) == 0):
        # RULES_TO_ASSERT = range(1, NUM_RULES+1)
        RULES_TO_ASSERT = []
        for r in range(1, NUM_RULES+1):
            RULES_TO_ASSERT.append(r)
    else:
        RULES_TO_ASSERT = rules

    print('Asserting rules: {}'.format(RULES_TO_ASSERT))

    # Initialise:
    results_clean = {"class_ids": results["class_ids"],
                         "rois": results["rois"],
                         "masks": results["masks"],
                         "scores": results["scores"]
                    }
    toDelete = np.zeros([results_clean["class_ids"].shape[0], 1])  # If 1, delete index

    # Note: RULE 1
    if (1 in RULES_TO_ASSERT):
        print('RULE 1 | Cleaning segmentation masks.')
        results_clean["masks"] = clean_masks(results_clean["masks"])

    # Remove instances without a segmentation mask:
    no_mask_idxs = np.where(np.sum(results_clean["masks"], axis=(0,1)) == 0)
    toDelete[no_mask_idxs] = 1
    results_clean = update_results(results_clean, toDelete)
    toDelete = np.zeros([results_clean["class_ids"].shape[0], 1])

    # Get overlapping pairs of masks (ignore overlaps with self):
    overlaps = utils.compute_overlaps_masks(results_clean["masks"], results_clean["masks"])
    np.fill_diagonal(overlaps, 0)
    overlap_pairs = get_overlap_pairs(overlaps)

    # If there are overlapping pairs:
    if(len(overlap_pairs) > 0):

        # Separate pairs into the same or different labels:
        same_labels = []
        diff_labels = []
        for idx0, idx1 in overlap_pairs:

            # Check labels:
            label0 = results_clean["class_ids"][idx0]
            label1 = results_clean["class_ids"][idx1]

            if(label0 == label1):
                same_labels.append((idx0, idx1))
            else:
                diff_labels.append((idx0, idx1))

        # Note: RULE 2
        if(2 in RULES_TO_ASSERT):

            for idx0, idx1 in same_labels:

                prob0 = results_clean["scores"][idx0]
                prob1 = results_clean["scores"][idx1]

                # Keep only the one with higher probability:
                if(overlaps[idx0, idx1] >= HIGH_OVERLAP):
                    if (prob0 > prob1):
                        print('RULE 2 | Pair: {} {} | Deleting {} with lower probability.'.format(idx0, idx1, idx1))
                        toDelete[idx1] = 1
                    else:
                        print('RULE 2 | Pair: {} {} | Deleting {} with lower probability.'.format(idx0, idx1, idx0))
                        toDelete[idx0] = 1

        # Note: RULES 4 - 5 (reflex)
        for idx0, idx1 in diff_labels:

            # Check if either have been deleted:
            if (toDelete[idx0] == 1 or toDelete[idx1] == 1):
                continue

            # Check labels:
            label0 = results_clean["class_ids"][idx0]
            label1 = results_clean["class_ids"][idx1]

            if(label0 == 5 or label1 == 5):

                # Set idx0 = reflex:
                if(label1 == 5):
                    temp = idx0
                    idx0 = idx1
                    idx1 = temp
                    label1 = label0

                # Check if it overlaps ulcers or white cells:
                if(label1 == 1 or label1 == 2):

                    # Get the probabilities:
                    prob0 = results_clean["scores"][idx0]
                    prob1 = results_clean["scores"][idx1]

                    # Note: RULE 4 (If the reflex is high probability, remove overlapping ulcers):
                    # Note: White cells are not removed here, because they may be overlapping other ulcers.
                    if(prob0 >= HIGH_PROB):

                        if(4 in RULES_TO_ASSERT):

                            if (label1 == 1):

                                if(overlaps[idx0, idx1] >= HIGH_OVERLAP):
                                    print('RULE 4 | Pair: {} {} | Deleting {} because it is most likely a reflex.'.format(idx0, idx1, idx1))
                                    toDelete[idx1] = 1

                    # Note: RULE 5 (If the reflex is low probability, it is probably an ulcer):
                    else:

                        if (5 in RULES_TO_ASSERT):

                            print('RULE 5 | Pair: {} {} | Switching reflex to ulcer.'.format(idx0, idx1))
                            results_clean["class_ids"][idx0] = 1

    # Update
    results_clean = update_results(results_clean, toDelete)
    toDelete = np.zeros([results_clean["class_ids"].shape[0], 1])
    overlaps = utils.compute_overlaps_masks(results_clean["masks"], results_clean["masks"])
    np.fill_diagonal(overlaps, 0)
    overlap_pairs = get_overlap_pairs(overlaps)

    # Note: RULE 7 (Only one pupil can exist)
    if(7 in RULES_TO_ASSERT):
        pupil_idxs = np.where(results_clean["class_ids"] == 6)[0]
        if(len(pupil_idxs) > 1):
            print('RULE 7 | Only one pupil can exist.')
            pupil_scores = results_clean["scores"][pupil_idxs]
            best_pupil_idx = pupil_idxs[np.argmax(pupil_scores)]
            for idx in pupil_idxs:
                if not (idx == best_pupil_idx):
                    toDelete[idx] = 1

    # Note: RULE 8 (Only one hypopyon can exist)
    if (8 in RULES_TO_ASSERT):
        hypopyon_idxs = np.where(results_clean["class_ids"] == 3)[0]
        if (len(hypopyon_idxs) > 1):
            print('RULE 8 | Only one hypopyon can exist.')
            hypopyon_scores = results_clean["scores"][hypopyon_idxs]
            best_hypopyon_idx = hypopyon_idxs[np.argmax(hypopyon_scores)]
            for idx in hypopyon_idxs:
                if not (idx == best_hypopyon_idx):
                    toDelete[idx] = 1

    # Note: RULE 2 (Because the changes in labels may cause new overlaps)
    if ((2.1 in RULES_TO_ASSERT) or (2.2 in RULES_TO_ASSERT)):

        for idx0, idx1 in overlap_pairs:

            # Check if either have been deleted:
            if (toDelete[idx0] == 1 or toDelete[idx1] == 1):
                continue

            # Check labels:
            label0 = results_clean["class_ids"][idx0]
            label1 = results_clean["class_ids"][idx1]

            if(label0 == label1):

                prob0 = results_clean["scores"][idx0]
                prob1 = results_clean["scores"][idx1]

                # Keep only the one with higher probability:
                if (2.1 in RULES_TO_ASSERT):
                    if (overlaps[idx0, idx1] >= HIGH_OVERLAP):
                        if (prob0 > prob1):
                            print('RULE 2 | Pair: {} {} | Deleting {} with lower probability.'.format(idx0, idx1,
                                                                                                      idx1))
                            toDelete[idx1] = 1
                        else:
                            print('RULE 2 | Pair: {} {} | Deleting {} with lower probability.'.format(idx0, idx1,
                                                                                                      idx0))
                            toDelete[idx0] = 1
                elif (2.2 in RULES_TO_ASSERT):
                    if (prob0 > prob1):
                        print('RULE 2 | Pair: {} {} | Deleting {} with lower probability.'.format(idx0, idx1, idx1))
                        toDelete[idx1] = 1
                    else:
                        print('RULE 2 | Pair: {} {} | Deleting {} with lower probability.'.format(idx0, idx1, idx0))
                        toDelete[idx0] = 1

    # Update
    results_clean = update_results(results_clean, toDelete)
    toDelete = np.zeros([results_clean["class_ids"].shape[0], 1])
    overlaps = utils.compute_overlaps_masks(results_clean["masks"], results_clean["masks"])
    np.fill_diagonal(overlaps, 0)
    overlap_pairs = get_overlap_pairs(overlaps)

    # Note: RULE 9 (Combine overlapping instances of the same type)
    if (9 in RULES_TO_ASSERT):

        # Initialise:
        h, w, _ = results_clean["masks"].shape
        instance_count = 0
        class_ids = []
        rois = []
        masks = []
        scores = []

        # Get the number of unique labels:
        classes = np.unique(results_clean["class_ids"])

        for c in classes:

            # Combine all of the same class:
            mask = np.zeros_like(results_clean["masks"][:, :, 0])
            idxs = np.where(results_clean["class_ids"] == c)[0]
            for i in idxs:
                mask += results_clean["masks"][:, :, i]
            mask = mask.astype(bool)

            # Separate components:
            labeled = skimage.measure.label(mask)
            cc = skimage.measure.regionprops(labeled)
            for n, region in enumerate(cc):
                mask_n = (labeled == cc[n].label)
                bbox_n = utils.extract_bboxes(mask_n[:, :, np.newaxis])[0]

                # Find which scores contribute:
                score_n = 0
                numcc_n = 0
                for i in idxs:
                    mask_i = results_clean["masks"][:, :, i]
                    mask_overlap = mask_n * mask_i
                    if(np.sum(mask_overlap) == np.sum(mask_i)):
                        score_n += results_clean["scores"][i]
                        numcc_n += 1
                score_n = score_n/numcc_n

                # Update:
                class_ids.append(c)
                rois.append(bbox_n)
                masks.append(mask_n)
                scores.append(score_n)
                instance_count += 1


        if(len(class_ids) == 0):
            results_clean = {"class_ids": np.array(class_ids),
                             "rois": np.array(rois),
                             "masks": np.empty([h, w, 0]),
                             "scores": np.array(scores)
                             }
        else:
            results_clean = {"class_ids": np.array(class_ids),
                             "rois": np.array(rois),
                             "masks": np.transpose(np.array(masks), (1,2,0)),
                             "scores": np.array(scores)
                             }

    # Update
    toDelete = np.zeros([results_clean["class_ids"].shape[0], 1])
    overlaps = utils.compute_overlaps_masks(results_clean["masks"], results_clean["masks"])
    np.fill_diagonal(overlaps, 0)
    overlap_pairs = get_overlap_pairs(overlaps)

    # Note: RULE 6 (White cells cannot exist alone)
    if (6 in RULES_TO_ASSERT):
        whitecells_idxs = np.where(results_clean["class_ids"] == 2)[0]
        ulcer_idxs = np.where(results_clean["class_ids"] == 1)[0]
        for idx in whitecells_idxs:
            overlap = np.sum(overlaps[idx, ulcer_idxs])
            if (overlap == 0):
                if(results_clean["scores"][idx] >= HIGH_PROB):
                    print('RULE 6 | Switching white cells to ulcer.')
                    results_clean["class_ids"][idx] = 1
                else:
                   print('RULE 6 | Deleting {} because white cells cannot exist alone.'.format(idx))
                   toDelete[idx] = 1

    # Update
    results_clean = update_results(results_clean, toDelete)
    toDelete = np.zeros([results_clean["class_ids"].shape[0], 1])
    overlaps = utils.compute_overlaps_masks(results_clean["masks"], results_clean["masks"])
    np.fill_diagonal(overlaps, 0)
    overlap_pairs = get_overlap_pairs(overlaps)

    # Note: RULE 3 (Checking ulcer and white cells size)
    if(3 in RULES_TO_ASSERT):

        for idx0, idx1 in overlap_pairs:

            # Check labels:
            label0 = results_clean["class_ids"][idx0]
            label1 = results_clean["class_ids"][idx1]

            if ((label0 == 1 and label1 == 2) or (label0 == 2 and label1 == 1)):

                # Set idx0 = ulcer, idx1 = white cells:
                if (label0 == 2):
                    temp = idx0
                    idx0 = idx1
                    idx1 = temp

                # Get the masks (at the end, the white cell mask should be the union of both):
                mask_ulcer = results_clean["masks"][:, :, idx0]
                mask_whitecells = results_clean["masks"][:, :, idx1]
                mask_combined = (mask_whitecells + mask_ulcer).astype(bool)
                bbox_combined = utils.extract_bboxes(mask_combined[:, :, np.newaxis])[0]

                results_clean["masks"][:, :, idx1] = mask_combined
                results_clean["rois"][idx1] = bbox_combined

    # Note: Rule 10 (Edema cannot exist alone and mask should be combined with ulcer / white cells)
    if (10 in RULES_TO_ASSERT):

        edema_idxs = np.where(results_clean["class_ids"] == 4)[0]
        temp_idx1 = results_clean["class_ids"] == 1
        temp_idx2 = results_clean["class_ids"] == 2
        ulcerwhitecells_idxs = np.where(np.any(np.stack([temp_idx1, temp_idx2]),axis=0))[0]
        for idx in edema_idxs:
            overlap = np.sum(overlaps[idx, ulcerwhitecells_idxs])
            if (overlap == 0):
                print('RULE 10 | Deleting {} because edema cannot exist alone.'.format(idx))
                toDelete[idx] = 1

        # Update
        results_clean = update_results(results_clean, toDelete)
        overlaps = utils.compute_overlaps_masks(results_clean["masks"], results_clean["masks"])
        np.fill_diagonal(overlaps, 0)

        edema_idxs = np.where(results_clean["class_ids"] == 4)[0]
        for idx in edema_idxs:

            # Find overlapping masks to combine with:
            overlap_idxs = np.where(overlaps[idx, :] > 0)[0]
            overlap_labels = results_clean["class_ids"][overlap_idxs]

            # Look for white cells first, otherwise look for ulcers:
            combine_idx = np.where(overlap_labels == 2)[0]
            if(len(combine_idx) == 0):
                combine_idx = np.where(overlap_labels == 1)[0]

            # If there is something that overlaps:
            if not (len(combine_idx) == 0):
                combine_idx = overlap_idxs[combine_idx][0]

                mask_edema = results_clean["masks"][:, :, idx]
                mask_ulcerwhitecells = results_clean["masks"][:, :, combine_idx]
                mask_combined = (mask_edema + mask_ulcerwhitecells).astype(bool)
                bbox_combined = utils.extract_bboxes(mask_combined[:, :, np.newaxis])[0]

                results_clean["masks"][:, :, idx] = mask_combined
                results_clean["rois"][idx] = bbox_combined

    # Update
    results_clean = update_results(results_clean, toDelete)

    return results_clean


def clean_predictions_blue(results, rules=[]):
    """
        Cleans predictions by asserting some rules:

        MASKS:

        1. Segmentation masks have to be one piece,
           i.e. holes filled, small components removed.

        LABELS:

        2. If multiple of the same biomarker have HIGH overlap,
           only the one with the highest probability is retained.

        3. Only one pupil can exist,
           only the one with the highest probability is retained, regardless of overlap.

        4. Combine overlapping instances with the same label

        results = output of model.detect()
        rules   = which rules to assert; if empty, all are asserted

        Notes:- Assumes masks have already been thresholded by call to either
                get_prediction_in_small_size() or get_prediction_in_original_size()
        """

    # Constants:
    NUM_RULES = 4
    HIGH_PROB = 0.8     # Threshold for which something is considered "high probability"
    HIGH_OVERLAP = 0.7  # Threshold for which something is considered to have "high overlap"

    # Check rules:
    if (len(rules) == 0):
        # RULES_TO_ASSERT = range(1, NUM_RULES+1)
        RULES_TO_ASSERT = []
        for r in range(1, NUM_RULES + 1):
            RULES_TO_ASSERT.append(r)
    else:
        RULES_TO_ASSERT = rules

    print('Asserting rules: {}'.format(RULES_TO_ASSERT))

    # Initialise:
    results_clean = {"class_ids": results["class_ids"],
                     "rois": results["rois"],
                     "masks": results["masks"],
                     "scores": results["scores"]
                     }
    toDelete = np.zeros([results_clean["class_ids"].shape[0], 1])  # If 1, delete index

    # Note: RULE 1
    if (1 in RULES_TO_ASSERT):
        print('RULE 1 | Cleaning segmentation masks.')
        results_clean["masks"] = clean_masks(results_clean["masks"])

    # Remove instances without a segmentation mask:
    no_mask_idxs = np.where(np.sum(results_clean["masks"], axis=(0, 1)) == 0)
    toDelete[no_mask_idxs] = 1
    results_clean = update_results(results_clean, toDelete)
    toDelete = np.zeros([results_clean["class_ids"].shape[0], 1])

    # Get overlapping pairs of masks (ignore overlaps with self):
    overlaps = utils.compute_overlaps_masks(results_clean["masks"], results_clean["masks"])
    np.fill_diagonal(overlaps, 0)
    overlap_pairs = get_overlap_pairs(overlaps)

    # If there are overlapping pairs:
    if (len(overlap_pairs) > 0):

        # Separate pairs into the same or different labels:
        same_labels = []
        diff_labels = []
        for idx0, idx1 in overlap_pairs:

            # Check labels:
            label0 = results_clean["class_ids"][idx0]
            label1 = results_clean["class_ids"][idx1]

            if (label0 == label1):
                same_labels.append((idx0, idx1))
            else:
                diff_labels.append((idx0, idx1))

        # Note: RULE 2
        if (2 in RULES_TO_ASSERT):

            for idx0, idx1 in same_labels:

                prob0 = results_clean["scores"][idx0]
                prob1 = results_clean["scores"][idx1]

                # Keep only the one with higher probability:
                if (overlaps[idx0, idx1] >= HIGH_OVERLAP):
                    if (prob0 > prob1):
                        print('RULE 2 | Pair: {} {} | Deleting {} with lower probability.'.format(idx0, idx1, idx1))
                        toDelete[idx1] = 1
                    else:
                        print('RULE 2 | Pair: {} {} | Deleting {} with lower probability.'.format(idx0, idx1, idx0))
                        toDelete[idx0] = 1


    # Update
    results_clean = update_results(results_clean, toDelete)
    toDelete = np.zeros([results_clean["class_ids"].shape[0], 1])
    overlaps = utils.compute_overlaps_masks(results_clean["masks"], results_clean["masks"])
    np.fill_diagonal(overlaps, 0)
    overlap_pairs = get_overlap_pairs(overlaps)

    # Note: RULE 3 (Only one pupil can exist)
    if (3 in RULES_TO_ASSERT):
        pupil_idxs = np.where(results_clean["class_ids"] == 3)[0]
        if (len(pupil_idxs) > 1):
            print('RULE 3 | Only one pupil can exist.')
            pupil_scores = results_clean["scores"][pupil_idxs]
            best_pupil_idx = pupil_idxs[np.argmax(pupil_scores)]
            for idx in pupil_idxs:
                if not (idx == best_pupil_idx):
                    toDelete[idx] = 1

    # Update
    results_clean = update_results(results_clean, toDelete)
    toDelete = np.zeros([results_clean["class_ids"].shape[0], 1])
    overlaps = utils.compute_overlaps_masks(results_clean["masks"], results_clean["masks"])
    np.fill_diagonal(overlaps, 0)
    overlap_pairs = get_overlap_pairs(overlaps)

    # Note: RULE 4 (Combine overlapping instances of the same type)
    if (4 in RULES_TO_ASSERT):

        # Initialise:
        h, w, _ = results_clean["masks"].shape
        instance_count = 0
        class_ids = []
        rois = []
        masks = []
        scores = []

        # Get the number of unique labels:
        classes = np.unique(results_clean["class_ids"])

        for c in classes:

            # Combine all of the same class:
            mask = np.zeros_like(results_clean["masks"][:, :, 0])
            idxs = np.where(results_clean["class_ids"] == c)[0]
            for i in idxs:
                mask += results_clean["masks"][:, :, i]
            mask = mask.astype(bool)

            # Separate components:
            labeled = skimage.measure.label(mask)
            cc = skimage.measure.regionprops(labeled)
            for n, region in enumerate(cc):
                mask_n = (labeled == cc[n].label)
                bbox_n = utils.extract_bboxes(mask_n[:, :, np.newaxis])[0]

                # Find which scores contribute:
                score_n = 0
                numcc_n = 0
                for i in idxs:
                    mask_i = results_clean["masks"][:, :, i]
                    mask_overlap = mask_n * mask_i
                    if (np.sum(mask_overlap) == np.sum(mask_i)):
                        score_n += results_clean["scores"][i]
                        numcc_n += 1
                score_n = score_n / numcc_n

                # Update:
                class_ids.append(c)
                rois.append(bbox_n)
                masks.append(mask_n)
                scores.append(score_n)
                instance_count += 1

        if (len(class_ids) == 0):
            results_clean = {"class_ids": np.array(class_ids),
                             "rois": np.array(rois),
                             "masks": np.empty([h, w, 0]),
                             "scores": np.array(scores)
                             }
        else:
            results_clean = {"class_ids": np.array(class_ids),
                             "rois": np.array(rois),
                             "masks": np.transpose(np.array(masks), (1, 2, 0)),
                             "scores": np.array(scores)
                             }

    return results_clean


def get_overlap_pairs(overlaps):
    """
    overlaps    = output of utils.compute_overlaps() or utils.compute_overlaps_masks()

    """

    overlap_idxs = np.array(np.where(overlaps > 0))

    if (overlap_idxs.shape[1] > 0):

        # Organise pairs:
        flip_idxs = np.where(overlap_idxs[0, :] > overlap_idxs[1, :])[0]
        overlap_idxs[:, flip_idxs] = np.flipud(overlap_idxs[:, flip_idxs])
        overlap_idxs = np.unique(overlap_idxs, axis=1)
        overlap_pairs = []
        for i in range(overlap_idxs.shape[1]):
            overlap_pairs.append(tuple(overlap_idxs[:, i]))

    else:

        overlap_pairs = []

    return overlap_pairs


def clean_masks(masks):
    """
    masks   = [HEIGHT, WIDTH, NUM_INSTANCES]

    """

    # Initialise:
    cleaned_masks = np.zeros_like(masks)

    for i in range(masks.shape[2]):

        # Initialise:
        max_area = 0
        max_idx = -1

        # Fill holes:
        m = masks[:, :, i]
        m = binary_fill_holes(m)

        # Keep largest component only:
        labeled = skimage.measure.label(m)
        cc = skimage.measure.regionprops(labeled)
        for n, region in enumerate(cc):
            if (region.area > max_area):
                max_area = region.area
                max_idx = n

        # Update mask:
        if(max_idx > -1):
            cleaned_masks[:,:,i] = (labeled == cc[max_idx].label)

    return cleaned_masks


def plot_validation_summary_metrics(metrics_dir):

    filename = os.path.join(metrics_dir, 'performance_metrics_summary.txt')

    with open(filename, 'r') as rf:
        rf.readline() # Skip the first line
        headings = rf.readline()
        headings = headings.split('\t')
        headings[-1] = headings[-1][:-1]
        values = []
        for line in rf:
            value = line.split('\t')[:-1]
            values.append(value)
        values = np.array(values).astype(np.float32)

    num_cols = values.shape[1]-1

    for i in range(num_cols):

        x = values[:,0]
        y = values[:,i+1]
        title = headings[i+1]

        pyplot.figure()
        pyplot.plot(x, y)
        pyplot.title('{} | max @ epoch {}'.format(title, int(x[np.argmax(y)])))
        pyplot.savefig(os.path.join(metrics_dir, '{}.png'.format(title)))
        pyplot.close()


# Modified from mrcnn.visualize:
def display_instances_ulcer_colorscheme(image, boxes, masks, class_ids,
                                        title="", figsize=(16, 16), ax=None,
                                        show_mask=True, show_bbox=True,
                                        blue_light=False):
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = pyplot.subplots(1, figsize=figsize)
        auto_show = True

    # Colors:
    if blue_light:
        colors = ['xkcd:bright magenta',    # epidefect
                  'xkcd:purply blue',       # reflex
                  'xkcd:black',             # pupil
                  'xkcd:orange']            # limbus
    else:
        colors = ['xkcd:bright magenta',    # ulcer
                  'xkcd:bright turquoise',  # white cells
                  'xkcd:bright yellow',     # hypopyon
                  'xkcd:white',             # edema
                  'xkcd:purply blue',       # reflex
                  'xkcd:black',             # pupil
                  'xkcd:orange']            # limbus

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):

        # Choose color
        color = colors[class_ids[i]-1]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color, linewidth=2)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        pyplot.show()


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image
