
def get_training_config():
    hyper_params = {}
    hyper_params["batch_size"] = 4 # number of images passed into the network, increasing this requires more GPU memory
    hyper_params["epochs"] = 40 # number of steps to run
    hyper_params["batches_per_epoch"] = 10 #5000 # number of batch_size inputs before an epoch is completed
    hyper_params["batches_per_val_run"] = 10 #1000 # number of batch_size validation checks before we're done with a single validation step after the training step in an epoch

    ## input image dimensions for ONet
    hyper_params["img_rows"] = 128
    hyper_params["img_cols"] = 256

    ## ratio of input training to testing images - only used if provided with one path to a folder of training data
    hyper_params["data_split_training"] = 0.8
    hyper_params["data_split_test"] = 1.0 - hyper_params["data_split_training"]

    ## custom parameters
    hyper_params["save_period"] = 101 # save the current model for every save_period number of epoches
    hyper_params["save_best"] = True # save the best model wrt the depth map loss and accuracy functions
    hyper_params["output_model_structure"] = True # use keras to write an image of the model to disk

    hyper_params["model_outputs"] = ["depth_map", "normal_map"]

    hyper_params["data_aug_params"] = {} ## various data augmentations which are applied in data_loader.py

    ## crop either the top-half or the bottom-half (but not both) for invariance to having the camera up-right and up-side-down
    hyper_params["data_aug_params"]["cut_half_top"] = True 
    hyper_params["data_aug_params"]["cut_half_top_prob"] = 0.80
    hyper_params["data_aug_params"]["cut_half_top_crop_min"] = 0.50
    hyper_params["data_aug_params"]["cut_half_top_crop_max"] = 0.99
    hyper_params["data_aug_params"]["cut_half_bot"] = True
    hyper_params["data_aug_params"]["cut_half_bot_prob"] = 0.80
    hyper_params["data_aug_params"]["cut_half_bot_crop_min"] = 0.50
    hyper_params["data_aug_params"]["cut_half_bot_crop_max"] = 0.99

    hyper_params["data_aug_params"]["rgb2bgr_prob"] = 0.5 # mix images with rgb and bgr to encourage network to learn more robust features
    hyper_params["data_aug_params"]["jpeg_degrade"] = 0.4 # degrade the image via jpeg compression to encourage network to learn more robust features

    return hyper_params
