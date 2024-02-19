from clearml import PipelineController, Task
from first_mlops_pipeline.preprocess_upload_cifar10 import (
    preprocess_and_upload_cifar10,
    save_preprocessed_data,
)
from first_mlops_pipeline.train_model import train_model
from first_mlops_pipeline.update_model import (
    archive_existing_model,
    cleanup_repo,
    clone_repo,
    commit_and_push,
    configure_ssh_key,
    ensure_archive_dir,
    update_model,
    update_weights,
)
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
    from first_mlops_pipeline.update_model import (
        archive_existing_model,
        clone_repo,
        commit_and_push,
        configure_ssh_key,
        ensure_archive_dir,
        cleanup_repo,
        update_model,
        update_weights,
    )
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
            "dataset_project": "CIFAR-10 Project",
            "dataset_name": "CIFAR-10 Raw",
        },
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
            "processed_dataset_project": "CIFAR-10 Project",
            "processed_dataset_name": "CIFAR-10 Preprocessed",
        },
        function_return=["processed_dataset_id"],
        helper_functions=[save_preprocessed_data],
        cache_executed_step=False,
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
        cache_executed_step=False,
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
        cache_executed_step=False,
    )

    # Step 5: Update Model in GitHub Repository
    pipeline.add_function_step(
        name="update_model_in_github",
        function=update_model,
        function_kwargs={
            "model_id": "${train_cifar10_model.model_id}",  # Use model_id from the train_model step
        },
        helper_functions=[
            configure_ssh_key,
            update_weights,
            ensure_archive_dir,
            commit_and_push,
            cleanup_repo,
            archive_existing_model,
            clone_repo,
        ],
        cache_executed_step=False,
    )

    pipeline.start_locally(run_pipeline_steps_locally=True)
    print("CIFAR-10 pipeline initiated. Check ClearML for progress.")
