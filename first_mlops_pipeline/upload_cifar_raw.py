import argparse
from tensorflow.keras.datasets import cifar10
from clearml import Dataset


def upload_cifar10_as_numpy(dataset_project, dataset_name):
    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    raw_dataset = Dataset.create(
        dataset_name=dataset_name, dataset_project=dataset_project
    )
    raw_dataset.add_samples(
        samples=[
            (train_images, "train_images.npy"),
            (train_labels, "train_labels.npy"),
            (test_images, "test_images.npy"),
            (test_labels, "test_labels.npy"),
        ],
        sample_names=["train_images", "train_labels", "test_images", "test_labels"],
    )
    raw_dataset.upload()
    raw_dataset.finalize()
    print(f"Raw CIFAR-10 dataset uploaded with ID: {raw_dataset.id}")
    return raw_dataset.id  # Ensure this function returns the dataset ID


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload CIFAR-10 Raw Data to ClearML")
    parser.add_argument(
        "--dataset_project",
        type=str,
        required=True,
        help="ClearML dataset project name",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="ClearML dataset name for raw data",
    )
    args = parser.parse_args()
    upload_cifar10_as_numpy(args.dataset_project, args.dataset_name)
