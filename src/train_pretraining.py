from datetime import datetime
import random

import torch
from torch.utils import tensorboard
from transformers import VideoMAEForPreTraining, VideoMAEImageProcessor, VideoMAEConfig
from tqdm import tqdm

from src.data_loader import DataLoader, TRAIN_BATCH_SIZE
from src.logger import setup_logger
from src.models.utils.TubeMaskGenerator import TubeMaskGenerator


NUM_EPOCHS = 50
DEVICE_NAME = "cuda:1"

device = torch.device(DEVICE_NAME)

session_id = f"pretraining-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
logger = setup_logger(session_id)

tensorboard_summary_writer = tensorboard.SummaryWriter(f"tensorboard-logs/{session_id}")


def training_epoch(epoch_index, dataset):
    # Init main model
    videomae_config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base")
    videomae_for_pretraining_model = VideoMAEForPreTraining(videomae_config).to(device)
    videomae_for_pretraining_optimizer = torch.optim.AdamW(videomae_for_pretraining_model.parameters(), lr=5e-5)

    # Init tube mask generator
    tube_mask_generator = TubeMaskGenerator(input_size=(16, 224, 224), mask_ratio=0.9)
    
    sample_indices = [i for i in range(len(dataset))]
    random.shuffle(sample_indices)
    
    epoch_total_training_loss = 0.0
    epoch_total_validation_loss = 100.0

    batch_iteration_num = 0
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

        training_batch_loss_value = loss.item()
        epoch_total_training_loss += training_batch_loss_value

        logger.info(f"training_batch_{sample_index}_loss={training_batch_loss_value}")

        tensorboard_summary_writer.add_scalar(
            f"epoch_training_batch_loss/epoch_{epoch_index}",
            training_batch_loss_value,
            batch_iteration_num
        )
        
        batch_iteration_num += 1
    
    logger.info(f"epoch_total_training_loss={epoch_total_training_loss}")
    logger.info(f"epoch_total_training_loss={epoch_total_validation_loss}")
    
    epoch_average_training_loss = epoch_total_training_loss / len(dataset)
    epoch_average_validation_loss = epoch_total_validation_loss / len(dataset)
    
    return epoch_average_training_loss, epoch_average_validation_loss


def main():
    dataset = DataLoader(
        "/home/chetan_madan/scratch/abhilaksh/datasets/focus-mae-dataset-new",
        "pretrain"
    )
    logger.info("Dataset loaded.")

    for epoch_index in range(NUM_EPOCHS):
        logger.info(f"Starting epoch {epoch_index}")
        epoch_average_training_loss, epoch_average_validation_loss = training_epoch(epoch_index, dataset)

        logger.info(f"epoch_average_training_loss={epoch_average_training_loss}")
        logger.info(f"epoch_average_validation_loss={epoch_average_validation_loss}")

        tensorboard_summary_writer.add_scalar("epoch_average_training_loss", epoch_average_training_loss, epoch_index)
        tensorboard_summary_writer.add_scalar("epoch_average_validation_loss", epoch_average_validation_loss, epoch_index)

        logger.info(f"Epoch {epoch_index} completed.")
    
    tensorboard_summary_writer.close()


if __name__ == "__main__":
    main()
