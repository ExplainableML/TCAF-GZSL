import h5py
import pickle
import glob
from tqdm import tqdm
import numpy as np
import os
import cv2
import utils_create_features_selavi
import json
import torch
which_dataset="VGGSound"
use_audio=False

dictionary_pkl, list_h5=utils_create_features_selavi.create_dict_features(which_dataset=which_dataset, use_audio=use_audio)
number_of_videos=0
save_root_path = "/mnt/selavi_unaveraged_fixed"
for name in tqdm(list_h5):
        with h5py.File(name, "r") as f:
            # List all groups

            a_group_key = list(f.keys())[0]

            b_group_key=list(f.keys())[1]

            # Get the data
            data = list(f[a_group_key])
            data_video=list(f[b_group_key])
            list_features=[]
            list_video_names=[]
            list_fps=[]
            number_of_videos+=len(data_video)
            for index in range(len(f[b_group_key])):
                current_video=f[b_group_key][index].decode("utf-8")
                list_video_names.append(current_video)
                current_data=f[a_group_key][index]

                all_features_dict=dictionary_pkl[current_video]

                if use_audio==True:
                    features_from_dict=all_features_dict[2].cpu().detach().numpy()
                else:
                    features_from_dict=all_features_dict[0].cpu().detach().numpy()
                mean_feature=np.mean(features_from_dict, axis=0)

                array_sum = np.sum(features_from_dict)
                array_has_nan = np.isnan(array_sum)
                if array_has_nan==True:
                    print("NAN")
                fps=25
                if features_from_dict.shape[0] == 0:
                    print("0 size", current_video)

                list_features.append(np.array(features_from_dict))
                list_fps.append(fps)

            split_path=name.split("/")[2:]
            new_path=save_root_path
            for element in split_path[:-1]:
                if element is split_path[-4]:
                    element=element+"_no_averaged"
                new_path+="/"+element

            try:
                os.makedirs(new_path)
            except:
                pass
            new_path+="/"+(split_path[-1].rsplit(".", 1)[0])

            new_concatenated_list={}
            new_concatenated_list['features']=list_features
            new_concatenated_list['video_names']=list_video_names
            new_concatenated_list['fps']=list_fps

            with open(new_path+".pkl", 'wb') as f:
                pickle.dump(new_concatenated_list, f)

print(number_of_videos)
        #with h5py.File(new_path, "w") as hf:
        #    hf.create_dataset("data", data=list_features)
        #    hf.create_dataset("video_urls", dtype="S80", data=list_video_names)
        #    hf.create_dataset("length_features", data=length_features)
#file_error.close()

#with open("/mnt/video_error.json","w") as outfile:
#    json.dump(dict_error, outfile)
