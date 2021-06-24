import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
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

def train_epoch(model, optimizer, data_loader, loss_history, loss_func):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        x = data.cuda()
        data = rearrange(x, 'b p h w c -> b p c h w').cuda()
        target = target.type(torch.LongTensor).cuda()
        print('train target:')
        print(target)
        pred = model(data.float())
        print('pred.shape')
        print(pred.shape)
        output = F.log_softmax(pred, dim=1)
        print('output.shape')
        print(output.shape)
        # loss = F.nll_loss(output, target)
        # output = model(data.float())
        print('train output:')
        print(output)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history, loss_func):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            x = data.cuda()
            data = rearrange(x, 'b p h w c -> b p c h w').cuda()
            target = target.type(torch.LongTensor).cuda()
            print('eval target:')
            print(target)
            output = F.log_softmax(model(data.float()), dim=1)
            # loss = F.nll_loss(output, target, reduction='sum')
            # output = model(data.float())
            loss = loss_func(output, target)
            _, pred = torch.max(output, dim=1)
            print('eval pred:')
            print(pred)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')