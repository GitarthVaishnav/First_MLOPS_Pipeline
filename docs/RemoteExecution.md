# Remotely Execute Steps:
## Prerequisites

- Python 3.9+
- AWS SageMaker Terminal or Notebook OR Google Colaboratory Notebook

## Instructions:
This is a very simple task compared to how you  would normally execute steps locally, as it involves a lot of complex steps.

## Step 1: Initialise a Remote Agent
### Option 1: In a Terminal (local or AWS):
1. Install ClearML: 
    ```
    pip install clearml
    ```
2. Install ClearML Agent: 
    ```
    pip install clearml-agent
    ```
3. Create new credentials:
    1. Go to your [ClearML WebApp **Settings**](https://app.clear.ml/settings/workspace-configuration).
    2. Under the **WORKSPACES** section, go to **App Credentials**, and click **+ Create new credentials**
    3. Copy your credentials
4. Initialise the ClearML Agent:

    ```
    clearml-agent init
    ```
    
    This will prompt you to input the following:
    - configuration from ClearML Server
    - ClearML Hosts configuration (just press enter as you will be using web hosted server)
    - Default Output URI (used to automatically store models and artifacts): `S` (we select ClearML Server)
    - Enter git username for repository cloning (this is to access private repositories)
    - Enter git personal access token for user (this is to access private repositories)
5. Create a Queue:
    1. Go to your [ClearML WebApp **Orchestration>Queues**](https://app.clear.ml/workers-and-queues/queues).
    1. Under the **Orchestration** section, go to **queues**, and click **+ New Queue** (you will find this button on the top right side of the page)
    1. Enter the queue name in `lower case` and remember it
    
6. Start the Agent:
    ```
    clearml-agent daemon --queue "queue_name" --detached
    ```

### Option 2: In a Notebook Environment (Google Colab or AWS):
Feel free to use this notebook: [ClearML_Agent_AWS_or_Google_Colab_Notebooks](https://github.com/GitarthVaishnav/First_Pipeline/blob/master/notebooks/ClearML_Agent_AWS_or_Google_Colab_Notebooks.ipynb)
1. Install all necessary packages:

    ```
    !pip install clearml
    ```
    ```
    !pip install clearml-agent
    ```

2. Export this environment variable, it makes Matplotlib work in headless mode, so it won't output graphs to the screen

    ```
    !export MPLBACKEND=TkAg
    ```

3. (OPTIONAL): Enter your github credentials (only for private repositories)
In order for the agent to pull your code, it needs access to your repositories. If these are private, you'll have to supply the agent with github credentials to log in. Github/Bitbucket will no longer let you log in using username/password combinations. Instead, you have to use a personal token, read more about it [here for Github](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) and [here for Bitbucket](https://support.atlassian.com/bitbucket-cloud/docs/app-passwords/)

    ```
    import os
    os.environ["CLEARML_AGENT_GIT_USER"] = "username"
    os.environ["CLEARML_AGENT_GIT_PASS"] = "personal access token"
    ```
4. Create new credentials:
    1. Go to your [ClearML WebApp **Settings**](https://app.clear.ml/settings/workspace-configuration).
    2. Under the **WORKSPACES** section, go to **App Credentials**, and click **+ Create new credentials**
    3. Copy your credentials
5. Set your ClearML Credentials
    -   Insert the credentials you created in Step 4.
    - If you aren't using the ClearML hosted server, make sure to modify the server variables.

    ```
    from clearml import Task

    web_server = 'https://app.clear.ml'
    api_server = 'https://api.clear.ml'
    files_server = 'https://files.clear.ml'
    access_key = 'paste your access key'
    secret_key = 'paste your secret key'

    Task.set_credentials(web_host=web_server,
                        api_host=api_server,
                        files_host=files_server,
                        key=access_key,
                        secret=secret_key
                        )
    ```

6. Create new queue:
    1. Go to your [ClearML WebApp **Orchestration>Queues**](https://app.clear.ml/workers-and-queues/queues).
    2. Under the **Orchestration** section, go to **queues**, and click **+ New Queue** (you will find this button on the top right side of the page)
    3. Enter the queue name in `lower case` and remember it

7. Run clearml-agent:
    ```
    !clearml-agent daemon --queue "queue_name" --detached
    ```
    Make sure to type the correct queue name.
8. Run clearml-agent:
    ```
    !clearml-agent daemon --queue "gitarth" --detached --stop
    ```
## Step 2: Execute the Tasks Remotely:
### Prerequisites:
1. Fork and Clone a Repository (use this repository to try)
2. Using Any IDE, make sure that clearml is initialised. Follow: [ClearML Setup](https://github.com/GitarthVaishnav/First_Pipeline/blob/master/docs/Clearml_Setup.md)

### Implement a Task:
1. Implement a task (one is already implemented for you - check: [upload_cifar_raw.py](https://github.com/GitarthVaishnav/First_Pipeline/blob/113cf6b2dd15ad5b1896fa78f437830e5f6582c4/first_mlops_pipeline/upload_cifar_raw.py))

    ```python
    task = Task.init(
        project_name=dataset_project,
        task_name="Dataset Upload",
        task_type=Task.TaskTypes.data_processing,
    )
    task.execute_remotely(queue_name="queue_name", exit_process=True)
    ```
    This line `task.execute_remotely(queue_name="queue_name", exit_process=True)` is very important to execute a task remotely. Once you initialise a task, please make sure to have this line in the scope of the task.
    
    Note: Replace  `"queue_name"` with the actual Queue Name.

### Execute a task:
1. Run this task for the first time:
    ```
    python first_mlops_pipeline/upload_cifar_raw.py --dataset_project TrialProject1 --dataset_name Cifar10RawData
    ```
    This is going to run this experiment in the ClearML Agent.

2. For Subsequent Experiments:
    - Use the ClearML UI by cloning the exisitng experiment.
    - Edit the configuration:
        - Change the github settings to pull from latest, or input the commit hash.
        - Change the path to the script incase a different script is to be run.
    - Edit the parameters:
        - Change the command line arguments
        - Incase of a different script, add or remove required command line arguments
    - Click Enque and select the queue to run this experiment remotely.


