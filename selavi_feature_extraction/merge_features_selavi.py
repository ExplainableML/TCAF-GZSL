import pickle
import glob
from tqdm import tqdm

path_to_search="/home/omercea19/akata-shared/omercea19/path_backup/"
path_to_dump="/home/omercea19/akata-shared/shared/avzsl/selavi/activity_dump_selavi.pkl"

save_data=[]

for filename in tqdm(glob.iglob(path_to_search + '**/*.pkl', recursive=True)):
    with open(filename, 'rb') as pickle_file:
        content = pickle.load(pickle_file)

    video=content[0][0]
    audio=content[1][0]
    labels=content[2][0]
    filenames=content[3][0]
    save_data.append([video, labels, audio, filenames])

with open(path_to_dump, 'wb') as handle:
    pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


