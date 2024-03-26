from clearml import PipelineController, Task

from first_mlops_pipeline.preprocess_upload_cifar10 import (
    preprocess_and_upload_cifar10,
    save_preprocessed_data,
)
from first_mlops_pipeline.upload_cifar_raw import (
    save_numpy_arrays,
    upload_cifar10_as_numpy,
)


def create_cifar10_data_pipeline(
    pipeline_name: str = "CIFAR-10 Data Pipeline",
    dataset_project: str = "CIFAR-10 Project",
    raw_dataset_name: str = "CIFAR-10 Raw",
    processed_dataset_name: str = "CIFAR-10 Preprocessed",
    queue_name: str = "gitarth"
):
    from clearml import PipelineController, Task

    from first_mlops_pipeline.preprocess_upload_cifar10 import (
        preprocess_and_upload_cifar10,
        save_preprocessed_data,
    )
    from first_mlops_pipeline.upload_cifar_raw import (
        save_numpy_arrays,
        upload_cifar10_as_numpy,
    )

    # Initialize a new pipeline controller task
    pipeline = PipelineController(
        name=pipeline_name,
        project=dataset_project,
        version="1.0",
        add_pipeline_tags=True,
        auto_version_bump=True,
        target_project=dataset_project,
    )

    # Add pipeline-level parameters with defaults from function arguments
    pipeline.add_parameter(name="dataset_project", default=dataset_project)
    pipeline.add_parameter(name="raw_dataset_name", default=raw_dataset_name)
    pipeline.add_parameter(
        name="processed_dataset_name", default=processed_dataset_name
    )

    # Step 1: Upload CIFAR-10 Raw Data
    pipeline.add_function_step(
        name="upload_cifar10_raw_data",
        function=upload_cifar10_as_numpy,
        function_kwargs={
            "dataset_project": "${pipeline.dataset_project}",
            "dataset_name": "${pipeline.raw_dataset_name}",
        },
        task_type=Task.TaskTypes.data_processing,
        task_name="Upload CIFAR-10 Raw Data",
        function_return=["raw_dataset_id"],
        helper_functions=[save_numpy_arrays],
        cache_executed_step=False,
    )

    # Step 2: Preprocess CIFAR-10 Data
    pipeline.add_function_step(
        name="preprocess_cifar10_data",
        function=preprocess_and_upload_cifar10,
        function_kwargs={
            "raw_dataset_id": "${upload_cifar10_raw_data.raw_dataset_id}",
            "processed_dataset_project": "${pipeline.dataset_project}",
            "processed_dataset_name": "${pipeline.processed_dataset_name}",
        },
        task_type=Task.TaskTypes.data_processing,
        task_name="Preprocess and Upload CIFAR-10",
        function_return=["processed_dataset_id"],
        helper_functions=[save_preprocessed_data],
        cache_executed_step=False,
    )

    # Start the pipeline
    pipeline.start(queue=queue_name)
    print("CIFAR-10 pipeline initiated. Check ClearML for progress.")


if __name__ == "__main__":
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Run CIFAR-10 Data Pipeline"
    )
    parser.add_argument(
        "--pipeline_name",
        type=str,
        default="CIFAR-10 Data Pipeline",
        help="Name of the pipeline",
    )
    parser.add_argument(
        "--dataset_project",
        type=str,
        default="CIFAR-10 Project",
        help="Project name for datasets",
    )
    parser.add_argument(
        "--raw_dataset_name",
        type=str,
        default="CIFAR-10 Raw",
        help="Name for the raw dataset",
    )
    parser.add_argument(
        "--processed_dataset_name",
        type=str,
        default="CIFAR-10 Preprocessed",
        help="Name for the processed dataset",
    )
    parser.add_argument(
        "--queue_name",
        type=str,
        required=True,
        help="ClearML queue name",
    )
    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    create_cifar10_data_pipeline(
        pipeline_name=args.pipeline_name,
        dataset_project=args.dataset_project,
        raw_dataset_name=args.raw_dataset_name,
        processed_dataset_name=args.processed_dataset_name,
        queue_name=args.queue_name,
    )
