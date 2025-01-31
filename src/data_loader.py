import copy
import json
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from transformers import VideoMAEImageProcessor

from src.utils import batchify


TRAIN_BATCH_SIZE = 4

# Init image processor
videomae_image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")  # Use VideoMAE pretrained processor


class DataLoader():
    def __init__(self, dataset_root_dir, mode):
        self.dataset_root_dir = dataset_root_dir
        self.mode = mode
        self.batch_size = TRAIN_BATCH_SIZE

        normalize_transform_for_image_processor = transforms.Normalize(
            mean=videomae_image_processor.image_mean,
            std=videomae_image_processor.image_std
        )
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize_transform_for_image_processor,
        ])

        self.load_data()

    def load_clips_from_video(self, video_path):
        video_capture = cv2.VideoCapture(video_path)

        frames = []
        while True:
            success, frame = video_capture.read()
            if not success:
                break
            
            frames.append(frame)
        
        transformed_clip_frames = [self.transform(frame) for frame in frames]
        clip_tensor = torch.stack(transformed_clip_frames)
        return clip_tensor
    
    def load_clip_from_video(self, video_path, clip_index, num_frames_per_clip=16):
        start_frame_index = clip_index * num_frames_per_clip
        # end_frame_index = clip_index * num_frames_per_clip - 1

        video_capture = cv2.VideoCapture(video_path)

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
        frames = []
        
        for _ in range(num_frames_per_clip):
            ret, frame = video_capture.read()
            if not ret:
                break
            frames.append(frame)
        
        if len(frames) < num_frames_per_clip:
            raise ValueError(f"Not enough frames in the requested clip, video_path={video_path}, clip_index={clip_index}")
        
        video_capture.release()

        transformed_clip_frames = [self.transform(frame) for frame in frames]
        clip_tensor = torch.stack(transformed_clip_frames)
        return clip_tensor
    
    def batchify_samples(self, all_videos_samples, num_frames_per_video=48, num_frames_per_clip=16):
        num_clips_per_video = num_frames_per_video // num_frames_per_clip

        batched_samples = []
        
        current_batch = []
        for i, video_sample in enumerate(all_videos_samples):
            for clip_index in range(num_clips_per_video):
                if len(current_batch) == self.batch_size:
                    batched_samples.append(copy.deepcopy(current_batch))
                    current_batch = []
                
                current_batch.append({
                    **video_sample,
                    'clip_index': clip_index,
                })
        
        if len(current_batch) == self.batch_size:
            batched_samples.append(copy.deepcopy(current_batch))
            current_batch = []
        
        for batch in batched_samples:
            if len(batch) != self.batch_size:
                print(num_clips_per_video)
                print(len(current_batch))
                print(self.batch_size)        
                print(len(all_videos_samples))
                print(len(batched_samples))
                raise ValueError("Batch size mismatch")
        
        return batched_samples

    def load_data_from_dir(self, dir_name, split_name):
        data_dir = os.path.join(self.dataset_root_dir, dir_name, split_name)
        data_videos_dir = os.path.join(data_dir, "videos")
        data_labels_file = os.path.join(data_dir, "labels.json")

        with open(data_labels_file, 'r', encoding="utf-8") as f:
            labels = json.load(f)
        
        all_videos_samples = []
        for label in labels:
            all_videos_samples.append({
                'video_path': os.path.join(data_videos_dir, label['filename']),
                'label': label['label'],
            })
        
        self.batched_samples = self.batchify_samples(all_videos_samples)

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
        return len(self.batched_samples)

    def __getitem__(self, index):
        batch_clip_tensors = []
        batch_labels = []
        
        for sample in self.batched_samples[index]:
            clip_tensor = self.load_clip_from_video(sample['video_path'], sample['clip_index'])
            batch_clip_tensors.append(clip_tensor)

            batch_labels.append(sample['label'])
        
        batched_clips_tensor = torch.stack(batch_clip_tensors, dim=0)
        
        return batched_clips_tensor, batch_labels
