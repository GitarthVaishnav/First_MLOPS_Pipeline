import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from clearml import Task, Dataset, Model


def evaluate_model(model_id, processed_dataset_id):
    task = Task.init(
        project_name="CIFAR-10 Classification", task_name="Model Evaluation"
    )

    # Fetch and load the trained model
    model = Model.get(model_id=model_id)
    model_path = model.get_local_copy()
    loaded_model = load_model(model_path)

    # Access dataset
    dataset = Dataset.get(dataset_id=processed_dataset_id)
    dataset_path = dataset.get_local_copy()

    data = np.load(f"{dataset_path}/cifar10.npz")
    test_images, test_labels = data["test_images"], data["test_labels"]

    test_images = test_images / 255.0
    test_labels = to_categorical(test_labels)

    loss, accuracy = loaded_model.evaluate(test_images, test_labels)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    predictions = np.argmax(loaded_model.predict(test_images), axis=1)
    cm = confusion_matrix(np.argmax(test_labels, axis=1), predictions)
    task.get_logger().report_confusion_matrix(cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.show()

    # Log confusion matrix
    task.get_logger().report_matplotlib_figure(
        title="Confusion Matrix", series="Evaluation", figure=plt, iteration=0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CIFAR-10 model and log results with ClearML."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="ClearML model ID for the trained model",
    )
    parser.add_argument(
        "--processed_dataset_id",
        type=str,
        required=True,
        help="ClearML processed dataset id",
    )
    args = parser.parse_args()

    evaluate_model(args.model_id, args.processed_dataset_id)
