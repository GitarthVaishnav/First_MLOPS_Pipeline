import argparse
import os

import numpy as np
from clearml import Dataset
from tensorflow.keras.datasets import cifar10


def save_numpy_arrays(data, labels, data_filename, labels_filename):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset
    from tensorflow.keras.datasets import cifar10

    np.save(data_filename, data)
    np.save(labels_filename, labels)


def upload_cifar10_as_numpy(dataset_project, dataset_name):
    import argparse
    import os

    import numpy as np
    from clearml import Dataset, Task
    from tensorflow.keras.datasets import cifar10

    task = Task.init(
        project_name=dataset_project,
        task_name="Dataset Upload",
        task_type=Task.TaskTypes.data_processing,
    )
    task.execute_remotely(queue_name="queue_name", exit_process=True)
    # Load CIFAR-10 data
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    print(f"Train images shape: {train_images.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    # Save the numpy arrays to files
    save_numpy_arrays(
        train_images, train_labels, "train_images_10.npy", "train_labels_10.npy"
    )
    save_numpy_arrays(
        test_images, test_labels, "test_images_10.npy", "test_labels_10.npy"
    )

    # Create a new ClearML dataset
    raw_dataset = Dataset.create(
        dataset_name=dataset_name, dataset_project=dataset_project
    )

    # Add the saved numpy files to the dataset
    raw_dataset.add_files("train_images_10.npy")
    raw_dataset.add_files("train_labels_10.npy")
    raw_dataset.add_files("test_images_10.npy")
    raw_dataset.add_files("test_labels_10.npy")

    # Upload the dataset to ClearML
    raw_dataset.upload()
    raw_dataset.finalize()

    # Clean up: Remove the numpy files after upload
    os.remove("train_images_10.npy")
    os.remove("train_labels_10.npy")
    os.remove("test_images_10.npy")
    os.remove("test_labels_10.npy")

    print(f"Raw CIFAR-100 dataset uploaded with ID: {raw_dataset.id}")
    return raw_dataset.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload CIFAR-100 Raw Data to ClearML")
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
