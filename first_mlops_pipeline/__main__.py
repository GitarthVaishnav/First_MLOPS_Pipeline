if __name__ == "__main__":
    from first_mlops_pipeline.pipeline import create_cifar10_pipeline

    create_cifar10_pipeline(
        epochs=10,  # Default number of training epochs
        pipeline_name="CIFAR-10 Processing and Training Pipeline",  # Name of the pipeline
        dataset_project="CIFAR-10 Project",  # Default project name for datasets
        raw_dataset_name="CIFAR-10 Raw",  # Default name for the raw dataset
        processed_dataset_name="CIFAR-10 Preprocessed",  # Default name for the processed dataset
        env_path="/Users/apple/Desktop/AI_Studio/Introduction_to_MLOPS/First_Pipeline/.env",  # Path to the environment variables file
        repo_url="git@github.com:GitarthVaishnav/Cifar10_SimpleFlaskApp.git",  # URL to the Git repository
        development_branch="development",  # Default branch for development
    )
