import argparse
import os
from clearml import Dataset, Task


### THIS NEEDS TO RUN LOCALLY ###
def upload_local_directory_as_dataset(
    dataset_project, dataset_name, dataset_path
):
    import argparse
    import os
    from clearml import Dataset, Task
    task = Task.init(
        project_name=dataset_project,
        task_name="Dataset Upload",
        task_type=Task.TaskTypes.data_processing,
    )

    # Ensure the dataset path is valid
    if not os.path.isdir(dataset_path):
        raise ValueError(
            f"The specified path '{dataset_path}' is not a directory or does not exist."
        )

    print(f"Uploading dataset from {dataset_path}")

    # Create a new ClearML dataset
    raw_dataset = Dataset.create(
        dataset_name=dataset_name, dataset_project=dataset_project
    )

    # Add the dataset directory to the dataset
    raw_dataset.add_files(dataset_path)

    # Upload the dataset to ClearML
    raw_dataset.upload()
    raw_dataset.finalize()

    print(f"Dataset uploaded with ID: {raw_dataset.id}")
    return raw_dataset.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Dataset Directory to ClearML")
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
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset directory"
    )
    args = parser.parse_args()
    upload_local_directory_as_dataset(
        args.dataset_project, args.dataset_name, args.dataset_path
    )
