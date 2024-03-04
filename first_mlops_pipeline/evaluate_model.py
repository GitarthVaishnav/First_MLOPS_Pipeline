import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from clearml import Dataset, Model, Task
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


def log_debug_images(task, images, true_labels, predictions, num_images=10):
    import argparse

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from clearml import Dataset, Model, Task
    from sklearn.metrics import confusion_matrix
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import to_categorical

    # Log a set number of test images with their predicted and true labels
    for i in range(num_images):
        plt.figure(figsize=(2, 2))
        plt.imshow(images[i])
        plt.title(f"True: {true_labels[i]}, Pred: {predictions[i]}")
        plt.axis("off")
        # Log the debug image to ClearML
        task.logger.report_matplotlib_figure(
            title="Debug Images", series=f"Image {i + 1}", figure=plt, iteration=0
        )
        plt.close()


def evaluate_model(model_id, processed_dataset_id, project_name):
    import argparse

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from clearml import Dataset, Model, Task
    from sklearn.metrics import confusion_matrix
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import to_categorical

    task = Task.init(
        project_name=project_name,
        task_name="Model Evaluation",
        task_type=Task.TaskTypes.testing,
    )
    task.execute_remotely(queue_name="queue_name", exit_process=True)

    # Fetch and load the trained model
    model = Model(model_id=model_id)
    model_path = model.get_local_copy()
    loaded_model = load_model(model_path)

    # Access dataset
    dataset = Dataset.get(dataset_id=processed_dataset_id)
    dataset_path = dataset.get_local_copy()

    # Load the numpy arrays from the dataset
    test_images = np.load(f"{dataset_path}/test_images_preprocessed.npy")
    test_labels = np.load(f"{dataset_path}/test_labels_preprocessed.npy")
    test_labels_categorical = to_categorical(test_labels)

    # Evaluate the model
    loss, accuracy = loaded_model.evaluate(test_images, test_labels_categorical)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # Generate predictions
    predictions = np.argmax(loaded_model.predict(test_images), axis=1)
    true_labels = (
        test_labels.flatten()
    )  # Assuming test_labels were saved in categorical format

    # Compute and log the confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    task.get_logger().report_matplotlib_figure(
        title="Confusion Matrix", series="Evaluation", figure=plt, iteration=0
    )
    plt.close()

    # Log debug images
    log_debug_images(task, test_images, true_labels, predictions, num_images=10)


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
