import random

from tqdm import tqdm

from src.data_loader import DataLoader


NUM_EPOCHS = 5


def training_epoch(dataset):
    sample_indices = [i for i in range(len(dataset))]
    random.shuffle(sample_indices)
    
    for sample_index in tqdm(sample_indices):
        sample = dataset[sample_index]
        clips, label = sample

        print(f"{len(clips)}, {len(clips[0])}, {clips[0][0].shape}, {label}")

        # print(sample)
        # print()
    pass


def main():
    dataset = DataLoader(
        "/home/chetan_madan/scratch/abhilaksh/datasets/focus-mae-dataset-new",
        "pretrain"
    )

    training_epoch(dataset)


if __name__ == "__main__":
    main()
