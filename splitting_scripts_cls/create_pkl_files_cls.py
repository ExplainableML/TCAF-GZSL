import json
import torch
import h5py
import argparse
import pickle
import glob
from tqdm import tqdm
import numpy as np
import os
import cv2
def create_dict_features(which_dataset, original_dataset_path, use_audio=False):
    '''
    This extracts a dictionary with video_name: all_other_data
    '''
    dictionary_pkl = {}
    if which_dataset == "UCF":
        with open(original_dataset_path, 'rb') as f:
            loaded_pickle = pickle.load(f)
            for element in loaded_pickle:
                video_name = element[3].split(".")[0]
                dictionary_pkl[video_name] = element
    elif which_dataset == "ActivityNet":
        list_of_pkl=glob.glob(original_dataset_path+"/*")
        for path_name in tqdm(list_of_pkl):
            with open(path_name, 'rb') as f:
                loaded_pickle = pickle.load(f)
                for element in loaded_pickle:
                    video_name = element[3].split(".")[0]
                    dictionary_pkl[video_name] = element


    elif which_dataset == "VGGSound":
        list_of_pkl = glob.glob(original_dataset_path+"/*")
        for path_name in tqdm(list_of_pkl):
            with open(path_name, 'rb') as f:
                loaded_pickle = pickle.load(f)
                for element in loaded_pickle:
                    video_name = element[3].split(".")[0]
                    dictionary_pkl[video_name] = element

    if use_audio == True:
        if which_dataset == "UCF":
            with open("splitting_scripts_cls/mapping_audio_files_cls_features_UCF.pickle", 'rb') as handle:
                list_h5 = pickle.load(handle)
        elif which_dataset == "ActivityNet":
            with open("splitting_scripts_cls/mapping_audio_files_cls_features_ActivityNet.pickle", 'rb') as handle:
                list_h5 = pickle.load(handle)
        elif which_dataset == "VGGSound":
            with open("splitting_scripts_cls/mapping_audio_files_cls_features_VGGSound.pickle", 'rb') as handle:
                list_h5 = pickle.load(handle)
    else:
        if which_dataset == "UCF":
            with open("splitting_scripts_cls/mapping_videos_files_cls_features_UCF.pickle", 'rb') as handle:
                list_h5 = pickle.load(handle)
        elif which_dataset == "ActivityNet":
            with open("splitting_scripts_cls/mapping_videos_files_cls_features_ActivityNet.pickle", 'rb') as handle:
                list_h5 = pickle.load(handle)
        elif which_dataset == "VGGSound":
            with open("splitting_scripts_cls/mapping_videos_files_cls_features_VGGSound.pickle", 'rb') as handle:
                list_h5 = pickle.load(handle)

    return dictionary_pkl, list_h5


def save_pickle_files(which_dataset, use_audio, root_path, original_dataset_path):

    dictionary_pkl, list_h5=create_dict_features(which_dataset=which_dataset, original_dataset_path=original_dataset_path, use_audio=use_audio)
    number_of_videos=0
    save_root_path = root_path
    for class_path, videos in tqdm(list_h5.items()):

        list_embeddings_videos=[]
        list_fps=[]
        number_of_videos+=len(videos)
        for video_item in videos:
            embedding_video=dictionary_pkl[video_item.decode("utf-8")]

            if use_audio == True:
                embedding_video = embedding_video[2]
            else:
                embedding_video = embedding_video[0]

            list_embeddings_videos.append(embedding_video)
            list_fps.append(25)

        concatenated_path= save_root_path + class_path

        new_concatenated_list = {}
        new_concatenated_list['features'] = list_embeddings_videos
        new_concatenated_list['video_names'] = videos
        new_concatenated_list['fps'] = list_fps

        split_path = concatenated_path.split("/")[1:]
        new_path=""
        for element in split_path[:-1]:
            if element is split_path[-4]:
                element = element + "_no_averaged"
            new_path += "/" + element

        try:
            os.makedirs(new_path)
        except:
            pass

        concatenated_path=new_path+"/"+split_path[-1].rsplit(".",1)[0]

        with open(concatenated_path + ".pkl", 'wb') as f:
            pickle.dump(new_concatenated_list, f)

    print("total_number_of_videos", number_of_videos)

parser = argparse.ArgumentParser(description='Create the datasets.')
parser.add_argument('--dataset_name', type=str,
                    help='The dataset name for which the features should be splitted')
parser.add_argument('--path_original_dataset', type=str, help='The path where the original dataset is stored')
parser.add_argument('--path_splitted_dataset', type=str, help="The path where the splitted dataset is stored")

args = parser.parse_args()

save_pickle_files(args.dataset_name, True,  args.path_splitted_dataset, args.path_original_dataset)
save_pickle_files(args.dataset_name, False, args.path_splitted_dataset, args.path_original_dataset)
