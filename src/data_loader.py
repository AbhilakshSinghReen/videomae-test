import json
import os

import cv2

from src.utils import batchify


class DataLoader():
    def __init__(self, dataset_root_dir, mode):
        self.dataset_root_dir = dataset_root_dir
        self.mode = mode

        self.load_data()

    def load_clips_from_video(self, video_path):
        video_capture = cv2.VideoCapture(video_path)

        frames = []
        while True:
            success, frame = video_capture.read()
            if not success:
                break

            frames.append(frame)
        
        clips = batchify(frames, 16)
        return clips
    
    def load_data_from_dir(self, dir_name, split_name):
        data_dir = os.path.join(self.dataset_root_dir, dir_name, split_name)
        data_videos_dir = os.path.join(data_dir, "videos")
        data_labels_file = os.path.join(data_dir, "labels.json")

        with open(data_labels_file, 'r', encoding="utf-8") as f:
            labels = json.load(f)
        
        self.samples = []
        for label in labels:
            self.samples.append({
                'video_path': os.path.join(data_videos_dir, label['filename']),
                'label': label['label'],
            })

    def load_data(self):
        if self.mode == "pretrain":
            self.load_data_from_dir("videos", "train")
        elif self.mode == "train":
            self.load_data_from_dir("downsampled-videos", "train")
        elif self.mode == "val":
            self.load_data_from_dir("downsampled-videos", "val")
        elif self.mode == "test":
            self.load_data_from_dir("downsampled-videos", "test")
        else:
            raise ValueError("Invalid Mode for DataLoader")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        clips = self.load_clips_from_video(sample['video_path'])

        return clips, sample['label']
