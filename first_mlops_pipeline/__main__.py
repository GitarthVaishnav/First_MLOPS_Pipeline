import argparse

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Run CIFAR-10 Processing and Training Pipeline")

    # Add arguments
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--pipeline_name', type=str, default="CIFAR-10 Processing and Training Pipeline", help='Name of the pipeline')
    parser.add_argument('--dataset_project', type=str, default="CIFAR-10 Project", help='Project name for datasets')
    parser.add_argument('--raw_dataset_name', type=str, default="CIFAR-10 Raw", help='Name for the raw dataset')
    parser.add_argument('--processed_dataset_name', type=str, default="CIFAR-10 Preprocessed", help='Name for the processed dataset')
    parser.add_argument('--env_path', type=str, default="/root/Repositories/First_MLOPS_Pipeline/.env", help='Path to the environment variables file')
    parser.add_argument('--repo_url', type=str, default="git@github.com:GitarthVaishnav/Cifar10_SimpleFlaskApp.git", help='URL to the Git repository')
    parser.add_argument('--development_branch', type=str, default="development", help='Default branch for development')

    # Parse the arguments
    args = parser.parse_args()

    # Import the create_cifar10_pipeline function inside the main block to avoid unnecessary imports when this script is imported as a module elsewhere
    from first_mlops_pipeline.pipeline import create_cifar10_pipeline

    # Call the function with the parsed arguments
    create_cifar10_pipeline(
        epochs=args.epochs,
        pipeline_name=args.pipeline_name,
        dataset_project=args.dataset_project,
        raw_dataset_name=args.raw_dataset_name,
        processed_dataset_name=args.processed_dataset_name,
        env_path=args.env_path,
        repo_url=args.repo_url,
        development_branch=args.development_branch,
    )
