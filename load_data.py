import torch
import torch.utils.data as data
import numpy as np
import os
import cv2
import pickle


def get_frames(filename, n_frames=1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = np.linspace(0, v_len - 1, n_frames + 1, dtype=np.int16)
    frame_dims = np.array([224, 224, 3])
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_dims[0], frame_dims[1]))
            frames.append(frame)
    v_cap.release()
    return frames, v_len


class DatasetProcessing(data.Dataset):
    def __init__(self, videos_path, framespath):
        super(DatasetProcessing, self).__init__()
        # List of all videos path
        video_list = []
        for root, dirs, files in os.walk(videos_path):
            for file in files:
                fullpath = os.path.join(root, file)
                if ('.mp4' in fullpath):
                    video_list.append(fullpath)
        self.video_list = np.asarray(video_list)
        self.framespath = framespath

    def __getitem__(self, index):
        # Ensure that the raw videos are in respective folders and folder name matches the output class label
        video_label = self.video_list[index].split('/')[-2]
        video_name = self.video_list[index].split('/')[-1]
        # pklFileName = os.path.splitext(video_name)[0]
        # with open(self.framespath + '/' + pklFileName + '.pkl', 'rb') as f:
        #     w_list = pickle.load(f)
        #
        # return w_list[0], w_list[1]

        video_frames, len_ = get_frames(self.video_list[index], n_frames = 7)
        video_frames = np.asarray(video_frames)
        video_frames = video_frames/255
        class_list = ['<OUTPUT_CLASS_LABEL_1>', '<OUTPUT_CLASS_LABEL_N>']
        class_id_loc = class_list.index(video_label)
        label = class_id_loc
        d = torch.as_tensor(np.array(video_frames).astype('float'))
        l = torch.as_tensor(np.array(label).astype('float'))
        return (d, l)

    def __len__(self):
        return self.video_list.shape[0]