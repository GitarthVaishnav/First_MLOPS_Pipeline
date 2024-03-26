import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
from clearml import Dataset, Task


def preprocess_image(image_path, size=(64, 64)):
    from PIL import Image
    import numpy as np
    """Resize an image and normalize its pixel values."""
    with Image.open(image_path) as img:
        img_resized = img.resize(size)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
    return img_array


def preprocess_and_upload_dataset(
    raw_dataset_id, processed_dataset_project, processed_dataset_name, queue_name
):
    import argparse
    import os
    from pathlib import Path
    import numpy as np
    from PIL import Image
    from clearml import Dataset, Task
    task = Task.init(
        project_name=processed_dataset_project,
        task_name="Dataset Preprocessing",
        task_type=Task.TaskTypes.data_processing,
    )
    task.execute_remotely(queue_name=queue_name, exit_process=True)

    # Access the raw dataset
    raw_dataset = Dataset.get(dataset_id=raw_dataset_id)
    raw_data_path = raw_dataset.get_local_copy()

    # Create directories for preprocessed data
    preprocessed_dir = Path("./preprocessed")
    preprocessed_dir.mkdir(exist_ok=True)

    # Process images
    for category in ["cat", "dog"]:
        category_path = Path(raw_data_path) / category
        (preprocessed_dir / category).mkdir(exist_ok=True)
        for image_file in category_path.iterdir():
            image_data = preprocess_image(image_file)
            np.save(
                preprocessed_dir / category / (image_file.stem + ".npy"), image_data
            )

    # Create a new dataset for the preprocessed images
    processed_dataset = Dataset.create(
        dataset_name=processed_dataset_name,
        dataset_project=processed_dataset_project,
        parent_datasets=[raw_dataset_id],
    )

    # Add the preprocessed images to the dataset
    processed_dataset.add_files(
        str(preprocessed_dir), local_base_folder=str(preprocessed_dir)
    )

    # Upload the dataset to ClearML
    processed_dataset.upload()
    processed_dataset.finalize()

    # Cleanup
    for item in preprocessed_dir.rglob("*"):
        item.unlink()
    preprocessed_dir.rmdir()

    print(f"Preprocessed dataset uploaded with ID: {processed_dataset.id}")
    return processed_dataset.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess and Upload Image Dataset to ClearML"
    )
    parser.add_argument(
        "--raw_dataset_id",
        type=str,
        required=True,
        help="ID of the raw dataset in ClearML",
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
    parser.add_argument(
        "--queue_name", type=str, required=True, help="ClearML queue name"
    )
    args = parser.parse_args()
    preprocess_and_upload_dataset(
        args.raw_dataset_id,
        args.processed_dataset_project,
        args.processed_dataset_name,
        args.queue_name,
    )
