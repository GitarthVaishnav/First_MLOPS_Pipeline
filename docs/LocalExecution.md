# Locally Execute Steps:
## Prerequisites

- Python 3.9+
- Poetry for Python package management

## Installation

1. Install Poetry:
   ```sh
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   OR (following for AWS)
   ```sh
   curl -sSL https://install.python-poetry.org | POETRY_HOME=/root/poetry python3 -
   ```
3. Clone the repo:
    ```sh
   git clone https://github.com/GitarthVaishnav/First_MLOPS_Pipeline.git

   cd First_MLOPS_Pipeline
   ```

4. Install dependencies using Poetry:
    ```sh 
   poetry install
   ```

5. Activate the Poetry shell:
    ```sh
   poetry shell
   ```

6. **IMPORTANT:** Edit the pipeline default parameters and make a repository for deployment weights to be stored in, configure a deploy key for it, save it in a file and put the path of the file in the .env file. Sample project here, please **fork**: [https://github.com/GitarthVaishnav/Cifar10_SimpleFlaskApp](https://github.com/GitarthVaishnav/Cifar10_SimpleFlaskApp) | DON'T push to this project, pipeline will FAIL.

7. Run the pipeline:
    ```sh
   python -m first_mlops_pipeline
   ```
