from datetime import datetime
import random

import numpy as np
import torch
from torch.utils import tensorboard
from transformers import VideoMAEForPreTraining, VideoMAEImageProcessor, VideoMAEConfig
from tqdm import tqdm

from src.data_loader import DataLoader, TRAIN_BATCH_SIZE
from src.logger import setup_logger
from src.models.utils.TubeMaskGenerator import TubeMaskGenerator
from src.masking.TubeMaskingGenerator import TubeMaskingGenerator


NUM_EPOCHS = 50
DEVICE_NAME = "cuda:1"

device = torch.device(DEVICE_NAME)

session_id = f"pretraining-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
logger = setup_logger(session_id)

tensorboard_summary_writer = tensorboard.SummaryWriter(f"tensorboard-logs/{session_id}")


def training_epoch(epoch_index, dataset):
    # Init tube masking generator
    tube_masking_generator = TubeMaskingGenerator((8, 14, 14), 0.75)

    # Init main model
    videomae_config = VideoMAEConfig.from_pretrained("MCG-NJU/videomae-base")
    videomae_for_pretraining_model = VideoMAEForPreTraining(videomae_config).to(device)
    videomae_for_pretraining_optimizer = torch.optim.AdamW(videomae_for_pretraining_model.parameters(), lr=5e-5)
    
    sample_indices = [i for i in range(len(dataset))]
    random.shuffle(sample_indices)
    
    epoch_total_training_loss = 0.0
    epoch_total_validation_loss = 100.0

    batch_iteration_num = 0
    for sample_index in tqdm(sample_indices):
        sample = dataset[sample_index]
        batched_clips_tensor, batch_labels = sample

        batched_clips_tensor = batched_clips_tensor.to(device)

        batch_masks = []
        for _i in range(TRAIN_BATCH_SIZE):
            batch_masks.append(tube_masking_generator())        
        batch_masks_ndarray = np.stack(batch_masks, axis=0)

        batched_masks_tensor = torch.from_numpy(batch_masks_ndarray)
        batched_masks_tensor = batched_masks_tensor > 0.5
        batched_masks_tensor = batched_masks_tensor.to(device)
        
        videomae_for_pretraining_optimizer.zero_grad()
        outputs = videomae_for_pretraining_model(
            pixel_values=batched_clips_tensor,
            bool_masked_pos=batched_masks_tensor
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
        print("")
    
    tensorboard_summary_writer.close()


if __name__ == "__main__":
    main()
