import matplotlib
matplotlib.use('agg')
print('matplotlib using {} backend'.format(matplotlib.get_backend()))

import os
import sys
import pickle
import time
import warnings
import numpy as np
from matplotlib import pyplot

import SLITNet_model as modellib
from SLITNet_utils import SLITNetDataset_white as Dataset
from SLITNet_utils import restrict_within_limbus
from SLITNet_utils import get_predictions_in_original_size
from SLITNet_utils import get_predictions_of_interest
from SLITNet_utils import clean_predictions_white as clean_predictions
from SLITNet_utils import calculate_performance_metrics, calculate_hausdorff_metric
from SLITNet_utils import display_instances_ulcer_colorscheme

# Settings:
MODE = "inference"


def main():

    ################################# USER INPUT #################################

    # Save directory:
    if (len(sys.argv) == 5):
        # If specified from terminal:
        MODEL_DIR = sys.argv[1]
        MODEL_SUBDIR = sys.argv[2]
        MODEL_NUM = int(sys.argv[3])
        K_FOLD = sys.argv[4]
    else:
        print('Wrong number of input arguments.')
        return

    # Other variables:
    THRESHOLD_CLASS = 0.5
    THRESHOLD_MASK = 0.5
    THRESHOLD_NMS = 0.5

    # Rules:
    RULE_NAME = 'standard'
    RULES = [1, 2, 3, 4, 6, 7, 8, 9, 10]

    # Data:
    PATH_TO_DATA = 'Datasets'
    TEST_FILENAME = os.path.join(PATH_TO_DATA, 'test_data_{}.mat'.format(K_FOLD))

    # IDs of interest:
    IDS_OF_INTEREST = [1, 2, 3, 4, 5, 7]

    ###############################################################################

    # Check existence:
    if not os.path.exists(os.path.join(MODEL_DIR, MODEL_SUBDIR)):
        print('The folder does not exist: {}'.format(os.path.join(MODEL_DIR, MODEL_SUBDIR)))
        return

    # Save folder:
    SAVE_FOLDER = os.path.join(MODEL_DIR, MODEL_SUBDIR, 'test',
                               'model_' + str(MODEL_NUM).zfill(4) + '_THRESHOLD_CLASS={}_{}'.format(THRESHOLD_CLASS, RULE_NAME))
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    else:
        print('The sub-folder for model_{} exists. '
              'Any existing data may be overwritten.'.format(str(MODEL_NUM).zfill(4)))

    # Config:
    CONFIG_FILENAME = os.path.join(MODEL_DIR, 'config.pickle')
    config = pickle.load(open(CONFIG_FILENAME, 'rb', pickle.HIGHEST_PROTOCOL))
    if (MODE == "inference"):
        config.set_to_inference_mode(class_threshold=THRESHOLD_CLASS, nms_threshold=THRESHOLD_NMS)
    config.display()

    # Number of classes:
    NUM_CLASSES = config.NUM_CLASSES - 1

    # Write files:
    def initialize_write_file(write_folder):
        write_filename = os.path.join(write_folder, 'performance_metrics.txt')
        with open(write_filename, 'a') as wf:
            wf.write('model_{}'.format(MODEL_NUM))
            wf.write('\n\tDSC')
            for i in range(NUM_CLASSES-1):
                wf.write('\t')
            wf.write('\tPred_Area')
            for i in range(NUM_CLASSES-1):
                wf.write('\t')
            wf.write('\tTruth_Area')
            for i in range(NUM_CLASSES-1):
                wf.write('\t')
            wf.write('\tHD')
            for i in range(NUM_CLASSES-1):
                wf.write('\t')
            wf.write('\nImage\t')
            for i in range(NUM_CLASSES):
                wf.write('Class_{}\t'.format(i + 1))
            for i in range(NUM_CLASSES):
                wf.write('Class_{}\t'.format(i + 1))
            for i in range(NUM_CLASSES):
                wf.write('Class_{}\t'.format(i + 1))
            for i in range(NUM_CLASSES):
                wf.write('Class_{}\t'.format(i + 1))
        return write_filename

    WRITE_FILENAME = initialize_write_file(SAVE_FOLDER)

    # Datasets:
    print('Test dataset: {}'.format(TEST_FILENAME))
    test_dataset = Dataset()
    test_dataset.load_dataset(TEST_FILENAME)
    test_dataset.prepare()
    print('Test dataset prepared: {} images'.format(test_dataset.num_images))

    # Ignore these specific warnings:
    warnings.filterwarnings("ignore", message="Anti-aliasing will be enabled by default in skimage 0.15 to")
    warnings.filterwarnings("ignore", message="Matplotlib is currently using agg")

    # Recreate the model in inference mode:
    model = modellib.MaskRCNN(mode=MODE,
                              config=config,
                              model_dir=os.path.join(MODEL_DIR, MODEL_SUBDIR))
    model_path = os.path.join(MODEL_DIR, MODEL_SUBDIR, 'mask_rcnn_ulcer_' + str(MODEL_NUM).zfill(4) + '.h5')
    model.load_weights(model_path,
                       by_name=True)
    print('Loading weights from: {}'.format(model_path))

    # Save images:
    for image_id in test_dataset.image_ids:

        # Load image and annotations:
        image_orig, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset=test_dataset,
                                                                                       config=config,
                                                                                       image_id=image_id,
                                                                                       use_mini_mask=False)
        # Run model detection:
        start_time = time.time()
        results = model.detect([image_orig])[0]
        elapsed_time = time.time() - start_time
        print('Image {}: {} seconds'.format(image_id, elapsed_time))

        ################################### DISPLAY ###########################################

        # Manual:
        ground_truth_orig = test_dataset.get_ground_truth_in_original_size(image_id)

        # Automatic:
        results_orig = get_predictions_in_original_size(image_meta, results, THRESHOLD_MASK)
        results_orig = restrict_within_limbus(results_orig, IDS_TO_RESTRICT=[1,2,4,6], LIMBUS_ID=7, HIGH_OVERLAP=0.7)
        results_orig = restrict_within_limbus(results_orig, IDS_TO_RESTRICT=[3], LIMBUS_ID=7, HIGH_OVERLAP=0.5)
        results_orig_cleaned = clean_predictions(results_orig, RULES)

        # Manual:
        ground_truth_interest = get_predictions_of_interest(ground_truth_orig, IDS_OF_INTEREST)
        display_instances_ulcer_colorscheme(image=test_dataset.load_image(image_id),
                                            boxes=ground_truth_interest["rois"],
                                            masks=ground_truth_interest["masks"],
                                            class_ids=ground_truth_interest["class_ids"],
                                            title=str(image_id).zfill(3) + ' | manual',
                                            show_mask=False,
                                            show_bbox=False)

        save_filename = os.path.join(SAVE_FOLDER, str(image_id).zfill(3) + '_manual.png')
        pyplot.savefig(save_filename)
        pyplot.close()

        # Automatic:
        results_orig_interest = get_predictions_of_interest(results_orig_cleaned, IDS_OF_INTEREST)
        display_instances_ulcer_colorscheme(image=test_dataset.load_image(image_id),
                                            boxes=results_orig_interest["rois"],
                                            masks=results_orig_interest["masks"],
                                            class_ids=results_orig_interest["class_ids"],
                                            title=str(image_id).zfill(3) + ' | auto',
                                            show_mask=False,
                                            show_bbox=False)

        save_filename = os.path.join(SAVE_FOLDER, str(image_id).zfill(3) + '_auto.png')
        pyplot.savefig(save_filename)
        pyplot.close()

        ################################### METRICS ###############################################

        def write_performance_metrics(write_filename, dsc_metrics, hd_metrics,
                                      predicted_area, truth_area):
            with open(write_filename, 'a') as wf:
                wf.write('\n{}\t'.format(image_id))
                for i in range(NUM_CLASSES):
                    wf.write('{}\t'.format(dsc_metrics[i, 0]))
                for i in range(NUM_CLASSES):
                    wf.write('{}\t'.format(predicted_area[i, 0]))
                for i in range(NUM_CLASSES):
                    wf.write('{}\t'.format(truth_area[i, 0]))
                for i in range(NUM_CLASSES):
                    wf.write('{}\t'.format(hd_metrics[i, 0]))

        # DSC:
        class_dice, pred_area, truth_area = calculate_performance_metrics(pred=results_orig_cleaned,
                                                                          truth=ground_truth_orig,
                                                                          num_classes=NUM_CLASSES)

        # HD:
        class_hd = calculate_hausdorff_metric(pred=results_orig_cleaned,
                                                           truth=ground_truth_orig,
                                                           num_classes=NUM_CLASSES)

        # Write to file:
        write_performance_metrics(WRITE_FILENAME, class_dice, class_hd, pred_area, truth_area)

    print('Finished.')


if __name__ == '__main__':
    main()
