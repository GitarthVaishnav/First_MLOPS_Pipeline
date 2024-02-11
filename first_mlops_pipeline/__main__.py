# Example usage, you can replace these with dynamic inputs or pipeline parameters
if __name__ == "__main__":
    from first_mlops_pipeline.pipeline import create_cifar10_pipeline

    create_cifar10_pipeline(
        epochs=10, pipeline_name="CIFAR-10 Processing and Training Pipeline"
    )
