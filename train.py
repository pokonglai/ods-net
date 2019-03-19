from __future__ import print_function

import os
import time

import sys
sys.path.append("..")

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

gpu_frac = 0.7
def get_session(gpu_fraction=gpu_frac):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB then gpu_frac=0.3 '''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

## attempt to limit the memory usage of TF
KTF.set_session(get_session(gpu_fraction=gpu_frac))


import keras
from keras.utils import plot_model
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np 
import cv2

import matplotlib.pyplot as plt

from utils import *

from get_config import *
from get_models import *
from data_loader import *


class ETATimer:
    ''' Simple class to store a timer for the ETA of one epoch/validation run '''
    def __init__(self, n_images):
        self.n_images = n_images
        self.all_times = []
        self.avg_time = 0

    def update(self, time_per_batch):
        self.all_times.append(time_per_batch)
        self.avg_time = np.mean(self.all_times)

    def get_eta(self, n_processed):
        return self.avg_time * (self.n_images - n_processed)

    def reset(self):
        self.all_times = []
        self.avg_time = 0       

def prep_data_for_network(img_array, hparams):
    # reshape the raw numpy data into what's needed for keras networks
    n_channels = 1
    if len(img_array.shape) == 4: # 4 implies --> batch size, rows, cols, channels
        n_channels = img_array.shape[3]

    if K.image_data_format() == 'channels_first': return img_array.reshape(img_array.shape[0], n_channels, hparams["img_rows"], hparams["img_cols"])
    else: return img_array.reshape(img_array.shape[0], hparams["img_rows"], hparams["img_cols"], n_channels)

class KerasModel_DataInBatches:
    def __init__(self, model_name, img_input_shape, hparams):
        self.model_name = model_name
        self.hyper_params = hparams
        self.img_input_shape = img_input_shape
        self.model = get_model(model_name, self.img_input_shape, hparams)
        if hparams["output_model_structure"]: plot_model(self.model, to_file=os.path.join("./model_vis/",self.model_name+'.png'), show_shapes=True)

    def train(self, input_datapaths, strOutputModelPath):

        train_start_time = time.time()

        ## create the log file
        log_file_train = open(os.path.join(strOutputModelPath, self.model_name+'_'+str(self.hyper_params["img_rows"])+"x"+str(self.hyper_params["img_cols"])+'_training.log'), 'w')
        log_file_val = open(os.path.join(strOutputModelPath, self.model_name+'_'+str(self.hyper_params["img_rows"])+"x"+str(self.hyper_params["img_cols"])+'_validation.log'), 'w')

        ## create the data batching generator functions
        fpaths_train, fpaths_val = input_datapaths
        batches_train = data_batcher(self.hyper_params["batch_size"], fpaths_train)
        batches_test = data_batcher(self.hyper_params["batch_size"], fpaths_val)

        ## loss function outputs use the layer name, when writing text to console this takes up too much space
        ## so create a mapping between the output layer name and a shorter keyword
        output_layers = ['depth_map', 'normal_map']
        output_layer_shortnames = {}
        output_layer_shortnames['depth_map'] = 'D'
        output_layer_shortnames['normal_map'] = 'N'

        ## do the same for the accuracy metrics
        acc_measures = ['rmse', 'median_dist', 'log10_error']
        acc_measures_shortnames = {}
        acc_measures_shortnames['rmse'] = 'rmse'
        acc_measures_shortnames['median_dist'] = 'med'
        acc_measures_shortnames['log10_error'] = 'log10'

        ## save the model with the best values for the depth map
        best_val_results = {}
        best_val_results['depth_map_loss'] = sys.maxsize
        best_val_results['depth_map_rmse'] = sys.maxsize
        best_val_results['depth_map_median_dist'] = sys.maxsize
        best_val_results['depth_map_log10_error'] = sys.maxsize

        ## write the header to log_file
        str_header = "epoch,"
        for l in output_layers:
            str_header += l+"_loss,"
            for am in acc_measures:
                str_header += l+"_"+am+","
        str_header+="\n"
        log_file_train.write(str_header)
        log_file_val.write(str_header)

        ndigits_iter = number_of_digits(self.hyper_params["epochs"])

        eta_train = ETATimer(self.hyper_params["batches_per_epoch"])
        eta_val = ETATimer(self.hyper_params["batches_per_val_run"])
        for i in range(self.hyper_params["epochs"]):
            str_iter_msg = str(i+1) + "/" + str(self.hyper_params["epochs"])
            iter_start_time  = time.time()

            ## store all the loss/acc values and average them as we work through an epoch
            latest_loss_train = {}
            batch_loss_train = {}
            latest_loss_val = {}
            batch_loss_val = {}
            for l in output_layers:
                latest_loss_train[l+"_loss"] = -1
                latest_loss_val[l+"_loss"] = -1
                batch_loss_train[l+"_loss"] = []
                batch_loss_val[l+"_loss"] = []

            latest_acc_train = {}
            batch_acc_train = {}
            latest_acc_val = {}
            batch_acc_val = {}
            for l in output_layers:
                for am in acc_measures:
                    latest_acc_train[l+"_"+am] = -1
                    latest_acc_val[l+"_"+am] = -1
                    batch_acc_train[l+"_"+am] = []
                    batch_acc_val[l+"_"+am] = []

            eta_train.reset()
            eta_val.reset()

            batch_start_time = time.time()
            for bs in range(self.hyper_params["batches_per_epoch"]):
                single_batch_start_time = time.time()

                training_imgs_fpaths = next(batches_train)

                ## left-right --> center depth and normals
                imgs_train_L, imgs_train_R, imgs_gt_depth, imgs_gt_norms = load_LR_norm_hdf5_batch(training_imgs_fpaths, self.hyper_params["data_aug_params"])
                imgs_L = prep_data_for_network(np.array(imgs_train_L), self.hyper_params)
                imgs_R = prep_data_for_network(np.array(imgs_train_R), self.hyper_params)
                imgs_D = prep_data_for_network(np.array(imgs_gt_depth), self.hyper_params)
                imgs_N = prep_data_for_network(np.array(imgs_gt_norms), self.hyper_params)
                x_train = [ imgs_L, imgs_R ]
                y_train = [ imgs_D, imgs_N ]

                ## train this batch
                fit_hist = self.model.fit(x_train, y_train, batch_size=self.hyper_params["batch_size"], epochs=1, verbose=0)

                ## record the training losses and accuracy
                for l in output_layers: 
                    batch_loss_train[l+"_loss"].append(fit_hist.history[l+"_loss"][0])
                    for am in acc_measures:
                        batch_acc_train[l+"_"+am].append(fit_hist.history[l+"_"+am][0])

                for l in output_layers:
                    latest_loss_train[l+"_loss"] = sum(batch_loss_train[l+"_loss"]) / len(batch_loss_train[l+"_loss"])
                    for am in acc_measures:
                        latest_acc_train[l+"_"+am] = sum(batch_acc_train[l+"_"+am]) / len(batch_acc_train[l+"_"+am])

                batch_end_time = time.time()
                batch_elapsed_time = batch_end_time - batch_start_time
                single_batch_elasped_time = batch_end_time - single_batch_start_time

                ## build the display string with the new updated average values for training loss and accuracy
                str_loss = ""
                str_acc = ""
                for l in output_layers:
                    str_loss += output_layer_shortnames[l]+"_loss=" + "{:5.6f}".format(latest_loss_train[l+"_loss"]) + " "
                    for am in acc_measures:
                        str_acc += output_layer_shortnames[l]+"_"+acc_measures_shortnames[am]+"=" +"{:5.6f}".format(latest_acc_train[l+"_"+am]) + " "

                # td, th, tm, ts = time_in_seconds_to_d_h_m_s(batch_elapsed_time)
                # str_time = str(int(th)) + "h - " + str(int(tm)) + "m - " + "{:3.2f}".format(ts)+"s        " # extra spaces to erase the extra chars at the end if the string is shorter

                # loss_msg = "  [ "+str_loss+"]+[ "+str_acc+"] >> time = " + str_time

                td, th, tm, ts = time_in_seconds_to_d_h_m_s(batch_elapsed_time) ## total elapsed time for training only
                str_time = str(int(th)) + "h - " + str(int(tm)) + "m - " + "{:3.2f}".format(ts)+"s" 

                eta_train.update(single_batch_elasped_time)
                eta_td, eta_th, eta_tm, eta_ts = time_in_seconds_to_d_h_m_s(eta_train.get_eta(bs+1) ) ## estimated time remaining
                str_eta_time = str(int(eta_th)) + "h - " + str(int(eta_tm)) + "m - " + "{:3.2f}".format(eta_ts)+"s"

                loss_msg = "  [ "+str_loss+"]+[ "+str_acc+"] >> time = " + str_time + " - ETA (" + str_eta_time + ")         " # extra spaces to erase the extra chars at the end if the string is shorter

                print_text_progress_bar(bs/(self.hyper_params["batches_per_epoch"]-1), bar_name="## T "+str_iter_msg + " >> ", bar_length=5, debug_msg=loss_msg)
            print()

            ## now load all of the validation data blocks and evaluate the network loss
            batch_start_time = time.time()
            for bs in range(self.hyper_params["batches_per_val_run"]):
                single_batch_start_time = time.time()

                validation_imgs_fpaths = next(batches_test)

                ## left-right --> center depth and normals
                imgs_test_L, imgs_test_R, imgs_gt_depth, imgs_gt_norms = load_LR_norm_hdf5_batch(validation_imgs_fpaths, self.hyper_params["data_aug_params"])
                imgs_L = prep_data_for_network(np.array(imgs_test_L), self.hyper_params)
                imgs_R = prep_data_for_network(np.array(imgs_test_R), self.hyper_params)
                imgs_D = prep_data_for_network(np.array(imgs_gt_depth), self.hyper_params)
                imgs_N = prep_data_for_network(np.array(imgs_gt_norms), self.hyper_params)
                x_test = [ imgs_L, imgs_R ]
                y_test = [ imgs_D, imgs_N ]

                ## evaluate this batch
                eval_score = self.model.evaluate(x_test, y_test, verbose=0)

                ## ONet.metrics_names --> ['loss', 'depth_map_loss', 'normal_map_loss', 'depth_map_rmse', 'normal_map_rmse']
                # print(self.model.metrics_names)
                # print(eval_score)
                for eval_idx in range(len(self.model.metrics_names)):
                    mname = self.model.metrics_names[eval_idx]
                    if mname in batch_loss_val: batch_loss_val[mname].append(eval_score[eval_idx])
                    elif mname in batch_acc_val: batch_acc_val[mname].append(eval_score[eval_idx])

                for l in output_layers:
                    latest_loss_val[l+"_loss"] = sum(batch_loss_val[l+"_loss"]) / len(batch_loss_val[l+"_loss"])
                    for am in acc_measures:
                        latest_acc_val[l+"_"+am] = sum(batch_acc_val[l+"_"+am]) / len(batch_acc_val[l+"_"+am])

                batch_end_time = time.time()
                batch_elapsed_time = batch_end_time - batch_start_time
                single_batch_elasped_time = batch_end_time - single_batch_start_time


                ## build the display string with the new updated average values for validation loss and accuracy
                str_loss = ""
                str_acc = ""
                for l in output_layers:
                    str_loss += output_layer_shortnames[l]+"_loss=" + "{:5.6f}".format(latest_loss_val[l+"_loss"]) + " "
                    for am in acc_measures:
                        str_acc += output_layer_shortnames[l]+"_"+acc_measures_shortnames[am]+"=" +"{:5.6f}".format(latest_acc_val[l+"_"+am]) + " "

                # td, th, tm, ts = time_in_seconds_to_d_h_m_s(batch_elapsed_time)
                # str_time = str(int(th)) + "h - " + str(int(tm)) + "m - " + "{:3.2f}".format(ts)+"s        " # extra spaces to erase the extra chars at the end if the string is shorter
                # acc_msg = "  [ "+str_loss+"]+[ "+str_acc+"] >> time = " + str_time


                td, th, tm, ts = time_in_seconds_to_d_h_m_s(batch_elapsed_time) ## total elapsed time for training only
                str_time = str(int(th)) + "h - " + str(int(tm)) + "m - " + "{:3.2f}".format(ts)+"s" 

                eta_val.update(single_batch_elasped_time)
                eta_td, eta_th, eta_tm, eta_ts = time_in_seconds_to_d_h_m_s(eta_val.get_eta(bs+1) ) ## estimated time remaining
                str_eta_time = str(int(eta_th)) + "h - " + str(int(eta_tm)) + "m - " + "{:3.2f}".format(eta_ts)+"s"

                acc_msg = "  [ "+str_loss+"]+[ "+str_acc+"] >> time = " + str_time + " - ETA (" + str_eta_time + ")         " # extra spaces to erase the extra chars at the end if the string is shorter

                print_text_progress_bar(bs/(self.hyper_params["batches_per_val_run"]-1), bar_name="$$ V "+str_iter_msg + " >> ", bar_length=5, debug_msg=acc_msg)
            print()

            ## display the total elapsed time for an iteration
            iter_end_time  = time.time()
            iter_elapsed_time = iter_end_time - iter_start_time
            td, th, tm, ts = time_in_seconds_to_d_h_m_s(iter_elapsed_time)
            str_time = str(int(th)) + "h - " + str(int(tm)) + "m - " + "{:3.2f}".format(ts)+"s        "
            print("Total elapsed time = " + str_time)
            print()

            ## log the training and valudation stats for this iteration
            log_line = str(i)+","
            for l in output_layers:
                log_line+=str(latest_loss_train[l+"_loss"])+","
                for am in acc_measures:
                    log_line+=str(latest_acc_train[l+"_"+am])+","
            log_line += "\n"
            log_file_train.write(log_line)
            log_file_train.flush()
            os.fsync(log_file_train.fileno())

            log_line = str(i)+","
            for l in output_layers:
                log_line+=str(latest_loss_val[l+"_loss"])+","
                for am in acc_measures:
                    log_line+=str(latest_acc_val[l+"_"+am])+","
            log_line += "\n"
            log_file_val.write(log_line)
            log_file_val.flush()
            os.fsync(log_file_val.fileno())

            ## making sure we save the best model for the depth map loss and accuracy measures
            if self.hyper_params["save_best"]:
                if best_val_results['depth_map_loss'] > latest_loss_val["depth_map_loss"]:
                    print("Lowest validation LOSS model = iter ", i)
                    best_val_results['depth_map_loss'] = latest_loss_val["depth_map_loss"]
                    self.save(strOutputModelPath, self.model_name+"_"+str(self.hyper_params["img_rows"])+"x"+str(self.hyper_params["img_cols"]) + "_lowest_loss")
                if best_val_results['depth_map_rmse'] > latest_acc_val["depth_map_rmse"]:
                    print("Lowest validation RMSE model = iter ", i)
                    best_val_results['depth_map_rmse'] = latest_acc_val["depth_map_rmse"]
                    self.save(strOutputModelPath, self.model_name+"_"+str(self.hyper_params["img_rows"])+"x"+str(self.hyper_params["img_cols"]) + "_lowest_rmse")
                if best_val_results['depth_map_median_dist'] > latest_acc_val["depth_map_median_dist"]:
                    print("Lowest validation MEDIAN DIST model = iter ", i)
                    best_val_results['depth_map_median_dist'] = latest_acc_val["depth_map_median_dist"]
                    self.save(strOutputModelPath, self.model_name+"_"+str(self.hyper_params["img_rows"])+"x"+str(self.hyper_params["img_cols"]) + "_lowest_med")
                if best_val_results['depth_map_log10_error'] > latest_acc_val["depth_map_log10_error"]:
                    print("Lowest validation LOG_10 model = iter ", i)
                    best_val_results['depth_map_log10_error'] = latest_acc_val["depth_map_log10_error"]
                    self.save(strOutputModelPath, self.model_name+"_"+str(self.hyper_params["img_rows"])+"x"+str(self.hyper_params["img_cols"]) + "_lowest_log10")
                print()

            ## we have a valid save_period, so check if it is time to save the current model to disk
            if self.hyper_params["save_period"] > 0:
                if (i % self.hyper_params["save_period"]) == 0:
                    self.save(strOutputModelPath, self.model_name+"_"+str(i).zfill(ndigits_iter)+"_"+str(self.hyper_params["img_rows"])+"x"+str(self.hyper_params["img_cols"]))

        log_file_train.close()
        log_file_val.close()

        train_end_time = time.time()
        td, th, tm, ts = time_in_seconds_to_d_h_m_s(train_end_time - train_start_time)
        str_time = str(int(td)) + "d - " + str(int(th)) + "h - " + str(int(tm)) + "m - " + "{:3.2f}".format(ts)+"s"
        print()
        print("Total training time for '" + self.model_name + "' ==> " + str_time)
        print()


    def save(self, folder_path, filename, show_debug_msg=False):
        full_path = os.path.join(folder_path, filename + ".h5")

        if show_debug_msg:
            print()
            print("*** Saving model to: " + full_path, end='   ', flush=True)

        self.model.save(full_path)

        if show_debug_msg:
            print(" ... DONE!!")
            print()



def train_with_multibatch():
    hyper_params = get_training_config()
    input_shape = (hyper_params["img_rows"], hyper_params["img_cols"], 3)

    str_model_name = "ODS_Net" ## regular SepUNet 
    # str_model_name = "ODS_Net_borderloss" ## with the Lpano loss function

    strTopFolder = "./data/" # input folder containing file lists
    fpaths_train = load_filelist(strTopFolder, "all_areas_train_v2.txt")
    fpaths_val = load_filelist(strTopFolder, "all_areas_val_v2.txt")

    strOutputModelPath = "./training_output/" # folder that will we dump log files and trained models which were saved according to save_period in hyper_params
    
    ## create the wrapper object that will take the folder of h5 images and feed it into the network in batches (as specified in hyper_params)
    kmodel = KerasModel_DataInBatches(str_model_name, input_shape, hyper_params)
    kmodel.train([fpaths_train, fpaths_val], strOutputModelPath) ## just LR -> L depth normals
    
    ## output final model
    kmodel.save(strOutputModelPath, str_model_name+"_"+str(hyper_params["img_rows"])+"x"+str(hyper_params["img_cols"]))

if __name__ == '__main__':
    train_with_multibatch()


