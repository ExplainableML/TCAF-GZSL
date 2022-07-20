import h5py
import pickle
import glob
from tqdm import tqdm
import numpy as np
import os
import cv2
def create_dict_features(which_dataset, use_audio=False):
    '''
    This extracts a dictionary with video_name: all_other_data
    '''
    dictionary_pkl = {}
    if which_dataset == "UCF":
        with open("/mnt/ucf_features_no_averaged_fixed", 'rb') as f:
            loaded_pickle = pickle.load(f)
            for element in loaded_pickle:
                video_name = element[3].split(".")[0]
                dictionary_pkl[video_name] = element
    elif which_dataset == "ActivityNet":
        list_of_pkl = glob.glob("/home/omercea19/akata-shared/shared/avzsl/data_supervised_full_videos/extracted_activity_features/no_averaged/*")
        for path_name in tqdm(list_of_pkl):
            with open(path_name, 'rb') as f:
                loaded_pickle = pickle.load(f)
                for element in loaded_pickle:
                    video_name = element[3].split(".")[0]
                    dictionary_pkl[video_name] = element


    elif which_dataset == "VGGSound":
        list_of_pkl = glob.glob("/home/omercea19/akata-shared/shared/avzsl/data_supervised_full_videos/extracted_vgg_features/no_averaged/*")
        for path_name in tqdm(list_of_pkl):
            with open(path_name, 'rb') as f:
                loaded_pickle = pickle.load(f)
                for element in loaded_pickle:
                    video_name = element[3].split(".")[0]
                    dictionary_pkl[video_name] = element

    if use_audio == True:
        if which_dataset == "UCF":
            list_h5 = glob.glob("/mnt/dat/UCF/features/supervised_ucf_all_videos/audio/*/*.h5")
        elif which_dataset == "ActivityNet":
            list_h5 = glob.glob("/mnt/dat/ActivityNet/features/supervised_activity_all_videos/audio/*/*.h5")
        elif which_dataset == "VGGSound":
            list_h5 = glob.glob("/mnt/dat/VGGSound/features/supervised_vgg_all_videos/audio/*/*.h5")
    else:
        if which_dataset == "UCF":
            list_h5 = glob.glob("/mnt/dat/UCF/features/supervised_ucf_all_videos/video/*/*.h5")
        elif which_dataset == "ActivityNet":
            list_h5 = glob.glob("/mnt/dat/ActivityNet/features/supervised_activity_all_videos/video/*/*.h5")
        elif which_dataset == "VGGSound":
            list_h5 = glob.glob("/mnt/dat/VGGSound/features/supervised_vgg_all_videos/video/*/*.h5")

    return dictionary_pkl, list_h5
