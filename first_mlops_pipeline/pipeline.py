from clearml import PipelineController, Task
from first_mlops_pipeline.preprocess_upload_cifar10 import (
    preprocess_and_upload_cifar10,
    save_preprocessed_data,
)
from first_mlops_pipeline.train_model import train_model
from first_mlops_pipeline.upload_cifar_raw import (
    upload_cifar10_as_numpy,
    save_numpy_arrays,
)
from first_mlops_pipeline.evaluate_model import evaluate_model, log_debug_images


def create_cifar10_pipeline(epochs: int, pipeline_name: str):
    from clearml import PipelineController, Task
    from first_mlops_pipeline.preprocess_upload_cifar10 import (
        preprocess_and_upload_cifar10,
    )
    from first_mlops_pipeline.train_model import train_model
    from first_mlops_pipeline.upload_cifar_raw import (
        upload_cifar10_as_numpy,
        save_numpy_arrays,
    )
    from first_mlops_pipeline.evaluate_model import evaluate_model, log_debug_images

    # Initialize a new pipeline controller task
    pipeline = PipelineController(
        name=pipeline_name,
        project="CIFAR-10 Project",
        version="1.0",
    )

    # Add pipeline-level parameters that can be configured
    pipeline.add_parameter(name="epochs", default=epochs)

    # Step 1: Upload CIFAR-10 Raw Data
    pipeline.add_function_step(
        name="upload_cifar10_raw_data",
        function=upload_cifar10_as_numpy,
        function_kwargs={
            "dataset_project": "CIFAR-10 Datasets",
            "dataset_name": "CIFAR-10 Raw",
        },
        function_return=["raw_dataset_id"],
        helper_functions=[save_numpy_arrays],
    )

    # Step 2: Preprocess CIFAR-10 Data
    pipeline.add_function_step(
        name="preprocess_cifar10_data",
        function=preprocess_and_upload_cifar10,
        function_kwargs={
            "raw_dataset_id": "${upload_cifar10_raw_data.raw_dataset_id}",
            "processed_dataset_project": "CIFAR-10 Datasets",
            "processed_dataset_name": "CIFAR-10 Preprocessed",
        },
        function_return=["processed_dataset_id"],
        helper_functions=[save_preprocessed_data],
    )

    # Step 3: Train Model
    pipeline.add_function_step(
        name="train_cifar10_model",
        function=train_model,
        function_kwargs={
            "processed_dataset_id": "${preprocess_cifar10_data.processed_dataset_id}",
            # Use the pipeline parameter directly in the function step
            "epochs": "${pipeline.epochs}",
        },
        function_return=["model_id"],
        helper_functions=[],
    )

    # Step 4: Evaluate Model
    pipeline.add_function_step(
        name="evaluate_cifar10_model",
        function=evaluate_model,
        function_kwargs={
            "model_id": "${train_cifar10_model.model_id}",
            "processed_dataset_id": "${preprocess_cifar10_data.processed_dataset_id}",
        },
        helper_functions=[log_debug_images],
    )

    pipeline.start_locally(run_pipeline_steps_locally=True)
    print("CIFAR-10 pipeline initiated. Check ClearML for progress.")
