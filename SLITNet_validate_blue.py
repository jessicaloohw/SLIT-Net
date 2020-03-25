import matplotlib
matplotlib.use('agg')
print('matplotlib using {} backend'.format(matplotlib.get_backend()))

import os
import sys
import pickle
import warnings
import numpy as np

import SLITNet_model as modellib
from SLITNet_utils import SLITNetDataset_blue as Dataset
from SLITNet_utils import restrict_within_limbus
from SLITNet_utils import get_predictions_in_small_size
from SLITNet_utils import clean_predictions_blue as clean_predictions
from SLITNet_utils import calculate_performance_metrics
from SLITNet_utils import plot_validation_summary_metrics

# Settings:
MODE = "inference"


def main():
    ########################################### USER INPUT ##########################################

    # Save directory:
    if (len(sys.argv) == 6):
        MODEL_DIR = sys.argv[1]
        MODEL_SUBDIR = sys.argv[2]
        MODEL_NUM_START = int(sys.argv[3])
        MODEL_NUM_END = int(sys.argv[4])
        K_FOLD = sys.argv[5]
    else:
        print('Wrong number of input arguments.')
        return

    # Other variables:
    THRESHOLD_CLASS = 0.5
    THRESHOLD_MASK = 0.5
    THRESHOLD_NMS = 0.5

    # Rules:
    RULE_NAME = 'standard'
    RULES = []

    # Data:
    PATH_TO_DATA = 'Datasets'
    TEST_FILENAME = os.path.join(PATH_TO_DATA, 'val_data_{}.mat'.format(K_FOLD))

    ################################################################################################

    # Check existence:
    if not os.path.exists(os.path.join(MODEL_DIR, MODEL_SUBDIR)):
        print('The folder does not exist: {}'.format(os.path.join(MODEL_DIR, MODEL_SUBDIR)))
        return

    # Config:
    CONFIG_FILENAME = os.path.join(MODEL_DIR, 'config.pickle')
    config = pickle.load(open(CONFIG_FILENAME, 'rb', pickle.HIGHEST_PROTOCOL))
    if (MODE == "inference"):
        config.set_to_inference_mode(class_threshold=THRESHOLD_CLASS, nms_threshold=THRESHOLD_NMS)
    config.display()

    # Number of classes:
    NUM_CLASSES = config.NUM_CLASSES - 1

    # Dataset:
    print('Validation dataset: {}'.format(TEST_FILENAME))
    test_dataset = Dataset()
    test_dataset.load_dataset(TEST_FILENAME)
    test_dataset.prepare()
    print('Validation dataset prepared: {} images'.format(test_dataset.num_images))

    # Ignore these specific warnings:
    warnings.filterwarnings("ignore", message="Anti-aliasing will be enabled by default in skimage 0.15 to")
    warnings.filterwarnings("ignore", message="Matplotlib is currently using agg")

    # Recreate the model in inference mode:
    model = modellib.MaskRCNN(mode=MODE,
                              config=config,
                              model_dir=os.path.join(MODEL_DIR, MODEL_SUBDIR))

    # Write file functions:
    def initialize_summary_file(write_folder):
        write_filename = os.path.join(write_folder, 'performance_metrics_summary.txt')
        with open(write_filename, 'a') as wf:
            wf.write('Rules: {}\nModel\t'.format(RULE_NAME))
            for i in range(NUM_CLASSES):
                wf.write('Class_{}\t'.format(i + 1))
            wf.write('Mean\tMean_1-4\tMean_1-2')
        return write_filename

    def initialize_write_file(write_folder, model_num):
        write_filename = os.path.join(write_folder, 'performance_metrics_model-{}.txt'.format(model_num))
        with open(write_filename, 'a') as wf:
            wf.write('Rules: {}\nImage\t'.format(RULE_NAME, model_num))
            for i in range(NUM_CLASSES):
                wf.write('Class_{}\t'.format(i + 1))
            wf.write('Mean\tMean_1-4\tMean_1-2')
        return write_filename

    def write_metrics(write_filename, performance_metrics, idx):
        with open(write_filename, 'a') as wf:
            wf.write('\n{}\t'.format(idx))
            for i in range(NUM_CLASSES):
                wf.write('{}\t'.format(performance_metrics[i, 0]))
            wf.write('{}\t{}\t{}\t'.format(np.mean(performance_metrics),
                                           np.mean(performance_metrics[:4]),
                                           np.mean(performance_metrics[:2])))

    # Save folder:
    SAVE_FOLDER = os.path.join(MODEL_DIR, MODEL_SUBDIR, 'val',
                               'performance_metrics_{}-{}_{}'.format(MODEL_NUM_START, MODEL_NUM_END, RULE_NAME))
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    SUMMARY_FILENAME = initialize_summary_file(SAVE_FOLDER)

    # Loop through models:
    for MODEL_NUM in range(MODEL_NUM_START, MODEL_NUM_END + 1):

        # Keep_track:
        summary_metrics = np.zeros([NUM_CLASSES, 1])
        summary_count = 0

        # Initialise write file:
        WRITE_FILENAME = initialize_write_file(SAVE_FOLDER, MODEL_NUM)

        # Load weights:
        model_path = os.path.join(MODEL_DIR, MODEL_SUBDIR, 'mask_rcnn_ulcer_' + str(MODEL_NUM).zfill(4) + '.h5')
        model.load_weights(model_path,
                           by_name=True)
        print('Loading weights from: {}'.format(model_path))

        for image_id in test_dataset.image_ids:

            print('Image {}'.format(image_id))

            # Load image and annotations:
            image_orig, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset=test_dataset,
                                                                                           config=config,
                                                                                           image_id=image_id,
                                                                                           use_mini_mask=False)
            # Run model detection:
            results = model.detect([image_orig])[0]

            # Get ground truth:
            ground_truth_small = {"class_ids": gt_class_id, "rois": gt_bbox, "masks": gt_mask}

            # Get cleaned predictions:
            results_small = get_predictions_in_small_size(results, THRESHOLD_MASK)
            results_small = restrict_within_limbus(results_small, IDS_TO_RESTRICT=[1, 3], LIMBUS_ID=4, HIGH_OVERLAP=0.7)
            results_small_cleaned = clean_predictions(results_small, RULES)

            # Calculate performance metrics:
            class_dice_small_cleaned, _, _ = calculate_performance_metrics(pred=results_small_cleaned,
                                                                           truth=ground_truth_small,
                                                                           num_classes=NUM_CLASSES)

            # Write performance metrics:
            write_metrics(WRITE_FILENAME, class_dice_small_cleaned, image_id)

            # Add to summary metrics:
            summary_metrics += class_dice_small_cleaned
            summary_count += 1

        # Calculate summary metrics:
        summary_metrics = summary_metrics / summary_count

        # Write summary metrics:
        write_metrics(SUMMARY_FILENAME, summary_metrics, MODEL_NUM)

    # Plot:
    plot_validation_summary_metrics(SAVE_FOLDER)

    print('Finished.')


if __name__ == "__main__":
    main()