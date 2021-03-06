from pathlib import Path
import os
import sys
sys.path.append("..")
from tqdm import tqdm
import torch
from c3d.c3d import C3D
import torchvision
import csv
import argparse
from csv import reader
import numpy as np
from timeit import default_timer as timer
from audioset_vggish_tensorflow_to_pytorch.vggish import VGGish
from audioset_vggish_tensorflow_to_pytorch.audioset import vggish_input, vggish_postprocess
import cv2
import pickle
from pydub import AudioSegment


parser=argparse.ArgumentParser(description="GZSL with ESZSL")
parser.add_argument('--index', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int)
args=parser.parse_args()



output_list_no_average=[]
output_list_average=[]

#vggish_extracted=np.load('/home/omercea19/akata-shared/datasets/vggsound/features/vggish_features_10s/people-marching/---g-f_I2yQ_000001.npy')

path=Path("/home/omercea19/akata-shared/datasets/vggsound/video")

dict_csv={}
list_classes=[]
count=0
with open('/home/omercea19/akata-shared/datasets/vggsound/metadata/vggsound.csv', 'r') as read_obj: # path of the metadata
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        key=str(row[0])+"_"+str(row[1])
        if row[2] not in list_classes:
            list_classes.append(row[2])
        dict_csv[key]=[row[2],row[3]]

list_classes.sort()
dict_classes_ids={}
for index,val in enumerate(sorted(list_classes)):
    dict_classes_ids[val]=index

list_of_files=[]
for f in tqdm(path.glob("**/*.mp4")):
        list_of_files.append(f)

chunk=int(len(list_of_files)/40)+1

list_of_files=list_of_files[args.index*chunk:(args.index+1)*chunk]


device=torch.cuda.set_device('cuda:'+str(args.gpu))
pytorch_model = VGGish()
pytorch_model.load_state_dict(torch.load('/home/omercea19/ExplainableAudioVisualLowShotLearning/audioset_vggish_tensorflow_to_pytorch/pytorch_vggish.pth')) # path of the vggish pretrained network
pytorch_model = pytorch_model.to(device)

pytorch_model.eval()
model=C3D().cuda()
model.load_state_dict(torch.load('/home/omercea19/ExplainableAudioVisualLowShotLearning/c3d.pickle'), strict=True)
model.eval()
counter=0

for f in tqdm(list_of_files):

    counter+=1
    if counter%3000==0:
        with open('/mnt/vggsound_features_averaged'+ str(args.index), 'wb') as handle:
            pickle.dump(output_list_average, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('/mnt/vggsound_features_no_averaged'+ str(args.index), 'wb') as handle:
            pickle.dump(output_list_no_average, handle, protocol=pickle.HIGHEST_PROTOCOL)

    mp4_version = AudioSegment.from_file(str(f), "mp4")
    mp4_version.export("/mnt/vggsound_dummy"+str(args.index)+".wav", format="wav")

    try:
        audio = torch.from_numpy(vggish_input.wavfile_to_examples("/mnt/vggsound_dummy"+str(args.index)+".wav"))
        audio=audio.float().to(device)
        audio = audio.unsqueeze(dim=1)
        vggish_output = pytorch_model(audio)
        vggish_output = vggish_output.detach().cpu().numpy()
        post_processor= vggish_postprocess.Postprocessor('/home/omercea19/ExplainableAudioVisualLowShotLearning/audioset_vggish_tensorflow_to_pytorch/vggish_pca_params.npz')
        vggish_output = post_processor.postprocess(vggish_output)
        vggish_output_average=np.average(vggish_output, axis=0)

        while True:
            cap = cv2.VideoCapture(str(f))
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video = torch.zeros((frameCount, frameHeight,frameWidth, 3), dtype=torch.float32)

            fc = 0
            ret = True

            while (fc < frameCount and ret):
                ret, image=cap.read()
                if ret==True:
                    torch_image=torch.from_numpy(image)
                    video[fc]=torch_image
                    fc += 1

            cap.release()
            if fc!=0:
                break

        list_clips=[]

        p= torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((128, 171)),
            torchvision.transforms.CenterCrop((112,112)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                             std=[0.22803, 0.22145, 0.216989]),

        ])
        for i in range(0,fc, 16):
            dummy = torch.zeros((16,3, 112, 112), dtype=torch.float32)
            clip=video[i:i+16]
            for j in range(clip.shape[0]):
                frame=clip[j]
                frame=frame.permute(2, 0, 1)
                frame=p(frame)
                dummy[j]=frame

            dummy=dummy.permute(1,0,2,3)
            dummy=((torch.unsqueeze(dummy,0)).float()).cuda()
            output=model(dummy)
            output=torch.squeeze(output)
            output=output.cpu().detach().numpy()
            list_clips.append(output)

        list_clips=np.array(list_clips)
        list_clips_average=np.average(list_clips, axis=0)

    except Exception as e:
        print(e)
        print(f)
        continue

    name_file=str(f).split("/")[-1]
    splitted_path=name_file.rsplit('_', 1)
    class_id=splitted_path[0]
    number=int(splitted_path[1].split(".")[0])
    search_name=class_id+"_"+str(number)
    class_name=dict_csv[search_name][0]
    class_id=dict_classes_ids[class_name]

    result_list=[list_clips, class_id, vggish_output, name_file]
    result_list_average=[list_clips_average, class_id, vggish_output_average, name_file]

    output_list_no_average.append(result_list)
    output_list_average.append(result_list_average)


with open('/mnt/vggsound_features_averaged'+str(args.index), 'wb') as handle:
    pickle.dump(output_list_average, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/mnt/vggsound_features_no_averaged'+str(args.index), 'wb') as handle:
    pickle.dump(output_list_no_average, handle, protocol=pickle.HIGHEST_PROTOCOL)

