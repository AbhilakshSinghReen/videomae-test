import random

import torch
import torchvision.transforms as transforms
from transformers import VideoMAEForPreTraining, VideoMAEImageProcessor, VideoMAEConfig
from tqdm import tqdm

from src.data_loader import DataLoader, TRAIN_BATCH_SIZE
from src.models.utils.TubeMaskGenerator import TubeMaskGenerator

import numpy as np


NUM_EPOCHS = 50
DEVICE_NAME = "cuda:1"

device = torch.device(DEVICE_NAME)


def training_epoch(dataset):
    # Init main model
    videomae_config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base")
    videomae_for_pretraining_model = VideoMAEForPreTraining(videomae_config).to(device)
    videomae_for_pretraining_optimizer = torch.optim.AdamW(videomae_for_pretraining_model.parameters(), lr=5e-5)

    # Init tube mask generator
    tube_mask_generator = TubeMaskGenerator(input_size=(16, 224, 224), mask_ratio=0.9)
    
    sample_indices = [i for i in range(len(dataset))]
    random.shuffle(sample_indices)
    
    epoch_total_loss = 0.0

    for sample_index in tqdm(sample_indices):
        sample = dataset[sample_index]
        batched_clips_tensor, batch_labels = sample

        batched_clips_tensor = batched_clips_tensor.to(device)

        num_frames = 16
        num_patches_per_frame = (videomae_config.image_size // videomae_config.patch_size) ** 2
        seq_length = (num_frames // videomae_config.tubelet_size) * num_patches_per_frame

        bool_masked_pos_1 = torch.randint(0, 2, (1, seq_length)).bool()
        bool_masked_pos_2 = torch.randint(0, 2, (1, seq_length)).bool()
        bool_masked_pos_3 = torch.randint(0, 2, (1, seq_length)).bool()
        bool_masked_pos_4 = torch.randint(0, 2, (1, seq_length)).bool()
        bool_masked_pos = torch.cat([
            bool_masked_pos_1,
            bool_masked_pos_1,
            bool_masked_pos_1,
            bool_masked_pos_1,
        ])

        size = (TRAIN_BATCH_SIZE, 1568)
        random_bool_tensor = torch.rand(size) > 0.5
        for b in range(0, TRAIN_BATCH_SIZE):
            for i in range(1411):
                random_bool_tensor[b][i] = True
            for i in range(1411, 1568):
                random_bool_tensor[b][i] = False
        
        random_bool_tensor = random_bool_tensor.to(device)

        # print(batched_clips_tensor.shape)
        # print(random_bool_tensor.shape)
        
        videomae_for_pretraining_optimizer.zero_grad()
        outputs = videomae_for_pretraining_model(
            pixel_values=batched_clips_tensor,
            bool_masked_pos=random_bool_tensor
        )
        loss = outputs.loss
        loss.backward()
        videomae_for_pretraining_optimizer.step()

        epoch_total_loss += loss.item()
        # print(loss)
        
        continue


        print(type(batch_clips_ndarray))
        print(batch_clips_ndarray.dtype)
        print(batch_clips_ndarray.shape)
        print(type(batch_labels))
        print(batch_labels)
        exit()
        continue

        for clip in clips:
            transformed_clip_frames = [transform(frame) for frame in clip]
            clip_tensor = torch.stack(transformed_clip_frames)
            # clip_tensor is of shape (16, 3, 224, 224)
            # we will add a batch dimension
            batched_clip_tensor = clip_tensor.unsqueeze(0)
            batched_clip_tensor = batched_clip_tensor.to(device)

            clip_mask = tube_mask_generator.generate_mask()
            clip_mask = np.concatenate([clip_mask[np.newaxis, ...] for _ in range(128)])
            # print(clip_mask.dtype)
            clip_mask_tensor = torch.from_numpy(clip_mask)
            batched_clip_mask_tensor = clip_mask_tensor.unsqueeze(0)
            batched_clip_mask_tensor = batched_clip_mask_tensor.to(device)

            # Create a random boolean tensor
            
            # random_bool_tensor[0][1] = True
            # random_bool_tensor[0][2] = True
            # random_bool_tensor[0][3] = True
            # random_bool_tensor[0][4] = True
            # random_bool_tensor[0][5] = True
            # random_bool_tensor[0][6] = True
            # random_bool_tensor[0][7] = True

            # print(clip_tensor.shape)
            # print(batched_clip_tensor.shape)
            # print(batched_clip_tensor.dtype)
            # # print(clip_mask_tensor.shape)
            # print(random_bool_tensor.shape)
            # print(random_bool_tensor.dtype)

            # print(videomae_config)
            # print(videomae_config.tubelet_size)
            # print(videomae_config.image_size)
            # print(videomae_config.patch_size)
            # sequence_length = (videomae_config.num_frames // videomae_config.tubelet_size) * (videomae_config.image_size // videomae_config.patch_size) ** 2.
            # print(sequence_length)
            # exit()

    epoch_average_loss = epoch_total_loss / len(dataset)
            
    
    return epoch_average_loss


def main():
    dataset = DataLoader(
        "/home/chetan_madan/scratch/abhilaksh/datasets/focus-mae-dataset-new",
        "pretrain"
    )
    print("Dataset loaded.")

    for epoch_index in range(NUM_EPOCHS):
        epoch_average_loss = training_epoch(dataset)

        print(f"Epoch {epoch_index + 1}, epoch_average_loss={epoch_average_loss}")


if __name__ == "__main__":
    main()
