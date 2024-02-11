import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
from clearml import Task, Dataset, OutputModel


class DebugImagesCallback(Callback):
    def __init__(self, task, validation_images, validation_labels, num_images=10):
        self.task = task
        self.validation_images = validation_images
        self.validation_labels = validation_labels
        self.num_images = num_images

    def on_epoch_end(self, epoch, logs=None):
        predictions = np.argmax(
            self.model.predict(self.validation_images[: self.num_images]), axis=1
        )
        for i, prediction in enumerate(predictions):
            plt.figure(figsize=(2, 2))
            plt.imshow(self.validation_images[i])
            plt.title(
                f"True: {np.argmax(self.validation_labels[i])}, Pred: {prediction}"
            )
            plt.axis("off")
            self.task.logger.report_matplotlib_figure(
                title=f"Debug Images - Epoch {epoch + 1}",
                series=f"Image {i + 1}",
                figure=plt,
                iteration=epoch,
            )
            plt.close()


def train_model(processed_dataset_id, epochs):
    task = Task.init(project_name="CIFAR-10 Classification", task_name="Model Training")

    # Access dataset
    dataset = Dataset.get(dataset_id=processed_dataset_id)
    dataset_path = dataset.get_local_copy()

    # Assuming the dataset is stored in .npz format
    data = np.load(f"{dataset_path}/cifar10.npz")
    train_images, train_labels = data["train_images"], data["train_labels"]
    test_images, test_labels = data["test_images"], data["test_labels"]

    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_labels, test_labels = to_categorical(train_labels), to_categorical(
        test_labels
    )

    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    debug_images_callback = DebugImagesCallback(
        task=task,
        validation_images=test_images,
        validation_labels=test_labels,
        num_images=10,
    )

    model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        validation_data=(test_images, test_labels),
        callbacks=[debug_images_callback],
    )

    model.save("model_cifar10.h5")

    # Assuming 'task' is your ClearML task
    output_model = OutputModel(task=task)
    output_model.update_weights(
        "model_cifar10.h5"
    )  # Upload the model weights to ClearML
    output_model.publish()  # Make sure the model is accessible
    task.upload_artifact("trained_model", artifact_object="model_cifar10.h5")
    return output_model.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a CNN on CIFAR-10 and log with ClearML."
    )
    parser.add_argument(
        "--processed_dataset_id",
        type=str,
        required=True,
        help="ClearML processed dataset id",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    args = parser.parse_args()

    train_model(args.processed_dataset_id, args.epochs)
