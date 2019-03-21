import sys
import os
import time
import random

import numpy as np 
import cv2

from keras import backend as K
from keras.models import load_model
from keras.metrics import *

from train import prep_data_for_network

import numpy as np 
import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt

from utils import *

from get_config import *
from get_models import *
from data_loader import *

from ply_io import *


def predict_from_hdf5_images():
    ''' Perform prediction using hdf5 images '''
    hyper_params = get_training_config()

    ## load the trained model

    ## border loss
    strModelFolder = "./trained_models/"
    strModelFilename = "SepUNet_Lpano.h5"

    strModelFullPath = os.path.join(strModelFolder, strModelFilename)
    custom_objs = {'berHu_loss_elementwise': berHu_loss_elementwise, 'berHu_loss_elementwise_w_border' : berHu_loss_elementwise_w_border, 'rmse' : rmse, 'median_dist' : median_dist, 'log10_error' : log10_error}
    model = load_model(strModelFullPath, custom_objects=custom_objs)
    model.summary()

    ## load the datalist
    strTopFolder = "G:/deep/ods/" # input folder containing file lists
    fpaths_train = load_filelist(strTopFolder, "all_areas_train_v2.txt", shuffle_flist=False)
    fpaths_val = load_filelist(strTopFolder, "all_areas_val_v2.txt", shuffle_flist=False)

    n_predictions = 20
    rng_indices = random.sample(range(0, len(fpaths_val)), n_predictions)

    for hdf5_idx in rng_indices:
        img_L, img_R, img_D, img_N = load_LR_norm_hdf5(fpaths_train[hdf5_idx], None) ## no need for data augmentations when performing predictions

        imgs_L = prep_data_for_network(np.array( [img_L] ), hyper_params) ## normalize those images!
        imgs_R = prep_data_for_network(np.array( [img_R] ), hyper_params)
        x_data = [ imgs_L, imgs_R ]
        y_data = img_D
        y_data[y_data <=0] = 0


        ## predict on the input image
        batch_input = x_data
        print("Predicting result ... ", end="", flush=True)
        pred_start_time = time.time()
        pred_result = model.predict(batch_input)
        pred_end_time = time.time()
        pred_dur = pred_end_time - pred_start_time
        print("Done! (" + str(pred_dur) + ") seconds")

        ## prediction results: - depth map is [0], normal map is [1]
        pred_00 = pred_result[0][0][:,:,0] 
        pred_normals = pred_result[1][0][:,:,0]

        pred_diff = np.abs(pred_00-y_data)
        print()
        print("  prediction stats ")
        print("----------------------")
        print("[Pred #0] max=", np.max(pred_00), ",  min=",np.min(pred_00), ",  mean=",np.mean(pred_00))
        print("[GT #0] max=", np.max(y_data), ",  min=",np.min(y_data), ",  mean=",np.mean(y_data))
        print("[ abs(Pred-GT) ] max=", np.max(pred_diff), ",  min=",np.min(pred_diff), ",  mean=",np.mean(pred_diff))
        print("----------------------")
        print() 

        ## filter noise - basically just extreme values
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        pred_00 = cv2.morphologyEx(pred_00, cv2.MORPH_OPEN, kernel)

        print()
        print("  prediction stats - after noise filtering")
        print("----------------------")
        print("[Pred #0] max=", np.max(pred_00), ",  min=",np.min(pred_00), ",  mean=",np.mean(pred_00))
        print("[GT #0] max=", np.max(y_data), ",  min=",np.min(y_data), ",  mean=",np.mean(y_data))
        print("[ abs(Pred-GT) ] max=", np.max(pred_diff), ",  min=",np.min(pred_diff), ",  mean=",np.mean(pred_diff))
        print("----------------------")
        print() 
        
        ## output prediction results along with lots of extra info
        min_val_0 = min(np.min(pred_00), np.min(y_data))
        max_val_0 = max(np.max(pred_00), np.max(y_data))
        strOutputPath = os.path.join("./prediction_results/HDF5_Images/", str(hdf5_idx))
        if not os.path.exists(strOutputPath): os.makedirs(strOutputPath)

        ## output depth maps with independantly scaled color maps
        plt.imsave(os.path.join(strOutputPath,'img_unscaled_pred.png'), pred_00, cmap='jet')
        plt.imsave(os.path.join(strOutputPath,'img_unscaled_gt.png'), y_data, cmap='jet')

        ## output depth maps with a single common color map 
        plt.imsave(os.path.join(strOutputPath,'img_scaled_pred.png'), pred_00, vmin=min_val_0, vmax=max_val_0, cmap='jet')
        plt.imsave(os.path.join(strOutputPath,'img_scaled_gt.png'), y_data, vmin=min_val_0, vmax=max_val_0, cmap='jet')

        ## output the left and right images used as input to the network
        plt.imsave(os.path.join(strOutputPath,'img_clr_L.png'), img_L)
        plt.imsave(os.path.join(strOutputPath,'img_clr_R.png'), img_R)


        plt.imsave(os.path.join(strOutputPath,'img_norm_gt.png'), ( (img_N+1.0)  *128).astype(np.uint8))
        plt.imsave(os.path.join(strOutputPath,'img_norm_pred.png'), ( (pred_normals+1.0)  *128).astype(np.uint8))


        


        ## output the point clouds for the prediction and the ground truth
        ## use the ground truth color image of the center camera to have the ideal alignment from color to depth map
        ptcloud_gt, npoints_gt = images_to_omni_pointcloud_equi_rectangular(img_L, img_D)
        ptcloud_pred, npoints_pred = images_to_omni_pointcloud_equi_rectangular(img_L, pred_00)
        output_pointcloud(npoints_gt, ptcloud_gt, os.path.join(strOutputPath, "ptcloud_gt"))
        output_pointcloud(npoints_pred, ptcloud_pred, os.path.join(strOutputPath, "ptcloud_pred"))

        ## textured meshes - prediction and ground truth
        texture_img_name = "image-texture"
        plt.imsave(os.path.join(strOutputPath, texture_img_name + '.png'), cv2.flip(img_L, 1 )) #

        ptcloud, triangles, texcoords, n_points_total = images_to_omni_textured_mesh_equi_rectangular(cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB), img_D)
        output_textured_mesh(ptcloud, texcoords, triangles, texture_img_name, "mesh_gt", strOutputPath)

        ptcloud, triangles, texcoords, n_points_total = images_to_omni_textured_mesh_equi_rectangular(cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB), pred_00)
        output_textured_mesh(ptcloud, texcoords, triangles, texture_img_name, "mesh_pred", strOutputPath)

    print(rng_indices)

def predict_from_video():
    ''' 
        Perform prediction on any ODS videos which are NxN in dimensions. 
        We assume the top half is the left eye and the bottom half is the right eye. 
    '''
    hyper_params = get_training_config()

    ## load the trained model first

    ## border loss
    strModelFolder = "./trained_models/"
    strModelFilename = "SepUNet_Lpano.h5"
    strModelFullPath = os.path.join(strModelFolder, strModelFilename)

    custom_objs = {'berHu_loss_elementwise': berHu_loss_elementwise, 'berHu_loss_elementwise_w_border' : berHu_loss_elementwise_w_border, 'rmse' : rmse, 'median_dist' : median_dist, 'log10_error' : log10_error}
    model = load_model(strModelFullPath, custom_objects=custom_objs)
    model.summary()

    ## set the top level input paths
    strVideoFolder = "./testing_videos/"
    strVideoName = "EpicUE4MatineeDemo.mp4"
    strVideoPath = os.path.join(strVideoFolder, strVideoName)

    ## output paths
    bOutputRGBDVideo = False ## determines whether or not an omni-directional rgb-d video of the results it output
    bOutputRGB_as_oriDims = False ## save color as the original size
    strOutputFolder = "./prediction_results/videos/EpicUE4MatineeDemo/"
    strOutputColorFolder = os.path.join(strOutputFolder, "color/")
    strOutputDepthFolder = os.path.join(strOutputFolder, "depth/")
    strOutputCombinedFolder = os.path.join(strOutputFolder, "combined/")
    if not os.path.exists(strOutputFolder): os.makedirs(strOutputFolder)
    if not os.path.exists(strOutputColorFolder): os.makedirs(strOutputColorFolder)
    if not os.path.exists(strOutputDepthFolder): os.makedirs(strOutputDepthFolder)
    if not os.path.exists(strOutputCombinedFolder): os.makedirs(strOutputCombinedFolder)

    
    ## load in the input video and process all the frames
    ## NOTE: the first prediction will be slow as all of the networks activations have to be created
    vcap = cv2.VideoCapture(strVideoPath)

    n_frame = 0
    while(vcap.isOpened()):
        ret, frame = vcap.read()
        if ret:
            rows, cols, chans = frame.shape

            img_L = frame[:rows//2]
            img_R = frame[rows//2:]

            img_L_resized = cv2.resize(img_L, (hyper_params["img_cols"], hyper_params["img_rows"]), interpolation=cv2.INTER_AREA)
            img_R_resized = cv2.resize(img_R, (hyper_params["img_cols"], hyper_params["img_rows"]), interpolation=cv2.INTER_AREA)

            imgs_L = prep_data_for_network(np.array( [cv2.cvtColor(img_L_resized, cv2.COLOR_BGR2RGB)/255] ), hyper_params)
            imgs_R = prep_data_for_network(np.array( [cv2.cvtColor(img_R_resized, cv2.COLOR_BGR2RGB)/255] ), hyper_params)
            x_data = [ imgs_L, imgs_R ]

            ## predict on the input images
            batch_input = x_data
            print("Predicting frame #"+str(n_frame)+" ... ", end="", flush=True)
            pred_start_time = time.time()
            pred_result = model.predict(batch_input)
            pred_end_time = time.time()
            pred_dur = pred_end_time - pred_start_time
            print("Done! (" + str(pred_dur) + ") seconds")
            
            ## prediction results: - depth map is [0], normal map is [1]
            pred_00 = pred_result[0][0][:,:,0] 
            pred_normals = pred_result[1][0][:,:,0]

            ## filter noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            pred_00 = cv2.morphologyEx(pred_00, cv2.MORPH_OPEN, kernel)

            ## colorize depth
            pred_00_normed = cv2.normalize(pred_00,None,0,1.0,cv2.NORM_MINMAX)
            pred_color = cv2.applyColorMap((pred_00_normed*255).astype(np.uint8), cv2.COLORMAP_JET)
            pred_on_left = cv2.addWeighted(img_L_resized, 0.7, pred_color, 0.3,0)

            ## because the input images are small (256w x 128h) lets upscale before displaying the results
            disp_upscale_amt = 3
            disp_img_L = cv2.resize(img_L_resized, None, fx=disp_upscale_amt, fy=disp_upscale_amt, interpolation=cv2.INTER_LINEAR )
            disp_img_R = cv2.resize(img_R_resized, None,  fx=disp_upscale_amt, fy=disp_upscale_amt, interpolation=cv2.INTER_LINEAR )
            disp_pred_color = cv2.resize(pred_color, None,  fx=disp_upscale_amt, fy=disp_upscale_amt, interpolation=cv2.INTER_LINEAR )
            disp_pred_on_left = cv2.resize(pred_on_left, None,  fx=disp_upscale_amt, fy=disp_upscale_amt, interpolation=cv2.INTER_LINEAR )

            ## build the final image grid of the input images, colorized depth map and depth map overlaid ontop of the left color image
            disp_top = np.concatenate((disp_img_L, disp_img_R), axis=1)
            disp_bot = np.concatenate((disp_pred_color, disp_pred_on_left), axis=1)
            disp_all = np.concatenate((disp_top, disp_bot), axis=0)
            cv2.imshow("Results", disp_all)

            ## output rgbd video
            if bOutputRGBDVideo:
                strFrame = str(n_frame).zfill(12)
                strOutputPathColor = os.path.join(os.path.join(strOutputFolder, "color/"), strFrame+".jpg")
                strOutputPathDepth = os.path.join(os.path.join(strOutputFolder, "depth/"), strFrame+".png")

                strOutputPathCombied = os.path.join(strOutputCombinedFolder, strFrame+".jpg")
                cv2.imwrite(strOutputPathCombied, disp_all)
                

                pred_16bit = (pred_00*1000).astype(np.uint16) ## prediction is in meters, store the depth in millimeters
                if bOutputRGB_as_oriDims:
                    cv2.imwrite(strOutputPathColor, img_L)
                    cv2.imwrite(strOutputPathDepth, pred_16bit)
                else:
                    cv2.imwrite(strOutputPathColor, img_L_resized)
                    cv2.imwrite(strOutputPathDepth, pred_16bit)

            n_frame+=1

            key_val = cv2.waitKey(1) & 0xFF
            if key_val == ord('q'):
                break
        else:
            break

    vcap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    # predict_from_hdf5_images()
    predict_from_video()
