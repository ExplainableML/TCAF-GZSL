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
        with open("/home/omercea19/akata-shared/shared/avzsl/selavi/ucf101-selavi_remove-1/ucf101_all_classes512_numsecaud1.pkl", 'rb') as f:
            loaded_pickle = pickle.load(f)
            for element in loaded_pickle:
                video_name = element[3][0]
                dictionary_pkl[video_name] = element
    elif which_dataset == "ActivityNet":
        with open("/home/omercea19/akata-shared/shared/avzsl/selavi/activity_dump_selavi.pkl",'rb') as f:
            loaded_pickle = pickle.load(f)
            for element in loaded_pickle:
                video_name = element[3][0]
                dictionary_pkl[video_name] = element


    elif which_dataset == "VGGSound":
        with open("/home/omercea19/akata-shared/shared/avzsl/selavi/vggsound_dump_selavi_full.pkl", 'rb') as f:
            loaded_pickle = pickle.load(f)
            for element in loaded_pickle:
                video_name = element[3][0]
                dictionary_pkl[video_name] = element

    if use_audio == True:
        if which_dataset == "UCF":
            list_h5 = glob.glob("/mnt/dat/UCF/features/self_supervised_split_nooverlap_1/audio/*/*.h5")
        elif which_dataset == "ActivityNet":
            list_h5 = glob.glob("/mnt/dat/ActivityNet/features/self_supervised_split_nooverlap/audio/*/*.h5")
        elif which_dataset == "VGGSound":
            list_h5 = glob.glob("/mnt/dat/VGGSound/features/self_supervised_split1_ex_others_nooverlap_respect_test_small_retrain/audio/*/*.h5")
    else:
        if which_dataset == "UCF":
            list_h5 = glob.glob("/mnt/dat/UCF/features/self_supervised_split_nooverlap_1/video/*/*.h5")
        elif which_dataset == "ActivityNet":
            list_h5 = glob.glob("/mnt/dat/ActivityNet/features/self_supervised_split_nooverlap/video/*/*.h5")
        elif which_dataset == "VGGSound":
            list_h5 = glob.glob("/mnt/dat/VGGSound/features/self_supervised_split1_ex_others_nooverlap_respect_test_small_retrain/video/*/*.h5")

    return dictionary_pkl, list_h5
