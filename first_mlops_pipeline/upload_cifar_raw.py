import argparse
import numpy as np
from tensorflow.keras.datasets import cifar10
from clearml import Dataset
import os


def save_numpy_arrays(data, labels, data_filename, labels_filename):
    import argparse
    import numpy as np
    from tensorflow.keras.datasets import cifar10
    from clearml import Dataset
    import os

    np.save(data_filename, data)
    np.save(labels_filename, labels)


def upload_cifar10_as_numpy(dataset_project, dataset_name):
    import argparse
    import numpy as np
    from tensorflow.keras.datasets import cifar10
    from clearml import Dataset
    import os

    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Save the numpy arrays to files
    save_numpy_arrays(
        train_images, train_labels, "train_images.npy", "train_labels.npy"
    )
    save_numpy_arrays(test_images, test_labels, "test_images.npy", "test_labels.npy")

    # Create a new ClearML dataset
    raw_dataset = Dataset.create(
        dataset_name=dataset_name, dataset_project=dataset_project
    )

    # Add the saved numpy files to the dataset
    raw_dataset.add_files("train_images.npy")
    raw_dataset.add_files("train_labels.npy")
    raw_dataset.add_files("test_images.npy")
    raw_dataset.add_files("test_labels.npy")

    # Upload the dataset to ClearML
    raw_dataset.upload()
    raw_dataset.finalize()

    # Clean up: Remove the numpy files after upload
    os.remove("train_images.npy")
    os.remove("train_labels.npy")
    os.remove("test_images.npy")
    os.remove("test_labels.npy")

    print(f"Raw CIFAR-10 dataset uploaded with ID: {raw_dataset.id}")
    return raw_dataset.id


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
