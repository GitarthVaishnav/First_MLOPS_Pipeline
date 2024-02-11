from clearml import PipelineController, Task


def create_cifar10_pipeline(epochs: int, pipeline_name: str):
    # Initialize a new pipeline controller task
    pipeline = PipelineController(
        name=pipeline_name,
        project="CIFAR-10 Project",
        version="1.0",
        description="Configurable pipeline from raw data upload to training and evaluation",
    )

    # Add pipeline-level parameters that can be configured
    pipeline.add_pipeline_parameter(name="epochs", value=epochs)

    # Step 1: Upload CIFAR-10 Raw Data
    pipeline.add_function_step(
        name="upload_cifar10_raw_data",
        function=upload_cifar10_as_numpy,
        function_kwargs={
            "dataset_project": "CIFAR-10 Datasets",
            "dataset_name": "CIFAR-10 Raw",
        },
        function_return=["raw_dataset_id"],
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
    )

    # Step 3: Train Model
    pipeline.add_function_step(
        name="train_cifar10_model",
        function=train_model,
        function_kwargs={
            "dataset_id": "${preprocess_cifar10_data.processed_dataset_id}",
            # Use the pipeline parameter directly in the function step
            "epochs": "${pipeline.epochs}",
        },
        function_return=["model_id"],
    )

    # Step 4: Evaluate Model
    pipeline.add_function_step(
        name="evaluate_cifar10_model",
        function=evaluate_model,
        function_kwargs={
            "model_id": "${train_cifar10_model.model_id}",
            "dataset_id": "${preprocess_cifar10_data.processed_dataset_id}",
        },
    )

    pipeline.start()
    print("CIFAR-10 pipeline initiated. Check ClearML for progress.")
