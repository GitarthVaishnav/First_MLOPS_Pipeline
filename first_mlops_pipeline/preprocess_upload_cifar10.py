import argparse
from clearml import Dataset
import numpy as np


def preprocess_and_upload_cifar10(
    raw_dataset_id, processed_dataset_project, processed_dataset_name
):
    raw_dataset = Dataset.get(dataset_id=raw_dataset_id)
    raw_data = raw_dataset.get_local_copy()
    train_images, train_labels = np.load(f"{raw_data}/train_images.npy"), np.load(
        f"{raw_data}/train_labels.npy"
    )
    test_images, test_labels = np.load(f"{raw_data}/test_images.npy"), np.load(
        f"{raw_data}/test_labels.npy"
    )
    train_images, test_images = train_images / 255.0, test_images / 255.0
    processed_dataset = Dataset.create(
        dataset_name=processed_dataset_name,
        dataset_project=processed_dataset_project,
        parent_datasets=[raw_dataset_id],
    )
    processed_dataset.add_samples(
        samples=[
            (train_images, "train_images.npy"),
            (train_labels, "train_labels.npy"),
            (test_images, "test_images.npy"),
            (test_labels, "test_labels.npy"),
        ],
        sample_names=["train_images", "train_labels", "test_images", "test_labels"],
    )
    processed_dataset.upload()
    processed_dataset.finalize()
    print(f"Preprocessed CIFAR-10 dataset uploaded with ID: {processed_dataset.id}")
    return processed_dataset.id  # Ensure this function returns the processed dataset ID


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess and Upload CIFAR-10 Data to ClearML"
    )
    parser.add_argument(
        "--raw_dataset_id",
        type=str,
        required=True,
        help="ID of the raw CIFAR-10 dataset in ClearML",
    )
    parser.add_argument(
        "--processed_dataset_project",
        type=str,
        required=True,
        help="ClearML project name for the processed dataset",
    )
    parser.add_argument(
        "--processed_dataset_name",
        type=str,
        required=True,
        help="Name for the processed dataset in ClearML",
    )
    args = parser.parse_args()
    preprocess_and_upload_cifar10(
        args.raw_dataset_id, args.processed_dataset_project, args.processed_dataset_name
    )
