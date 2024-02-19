import os
import datetime
import shutil
import argparse
from git import Repo, GitCommandError
from clearml import Model
from dotenv import load_dotenv


def configure_ssh_key(DEPLOY_KEY_PATH):
    import os
    import datetime
    import shutil
    import argparse
    from git import Repo, GitCommandError
    from clearml import Model
    from dotenv import load_dotenv

    """Configure Git to use the SSH deploy key for operations."""
    os.environ["GIT_SSH_COMMAND"] = f"ssh -i {DEPLOY_KEY_PATH} -o IdentitiesOnly=yes"


def clone_repo(REPO_URL, branch, DEPLOY_KEY_PATH):
    import os
    import datetime
    import shutil
    import argparse
    from git import Repo, GitCommandError
    from clearml import Model
    from dotenv import load_dotenv

    """Clone the repository."""
    configure_ssh_key(DEPLOY_KEY_PATH)
    repo_path = REPO_URL.split("/")[-1].split(".git")[0]
    try:
        repo = Repo.clone_from(REPO_URL, repo_path, branch=branch, single_branch=True)
        return repo, repo_path
    except GitCommandError as e:
        print(f"Failed to clone repository: {e}")
        exit(1)


def ensure_archive_dir(repo):
    import os
    import datetime
    import shutil
    import argparse
    from git import Repo, GitCommandError
    from clearml import Model
    from dotenv import load_dotenv

    """Ensures the archive directory exists within weights."""
    archive_path = os.path.join(repo.working_tree_dir, "weights", "archive")
    os.makedirs(archive_path, exist_ok=True)


def archive_existing_model(repo):
    import os
    import datetime
    import shutil
    import argparse
    from git import Repo, GitCommandError
    from clearml import Model
    from dotenv import load_dotenv

    """Archives existing model weights."""
    weights_path = os.path.join(repo.working_tree_dir, "weights")
    model_file = os.path.join(weights_path, "model.h5")
    if os.path.exists(model_file):
        today = datetime.date.today().strftime("%Y%m%d")
        archived_model_file = os.path.join(weights_path, "archive", f"model-{today}.h5")
        os.rename(model_file, archived_model_file)


def update_weights(repo, model_path):
    import os
    import datetime
    import shutil
    import argparse
    from git import Repo, GitCommandError
    from clearml import Model
    from dotenv import load_dotenv

    """Updates the model weights in the repository."""
    weights_path = os.path.join(repo.working_tree_dir, "weights")
    ensure_archive_dir(repo)
    archive_existing_model(repo)
    target_model_path = os.path.join(weights_path, "model.h5")
    shutil.move(model_path, target_model_path)  # Use shutil.move for cross-device move
    repo.index.add([target_model_path])


def commit_and_push(repo, model_id, DEVELOPMENT_BRANCH):
    import os
    import datetime
    import shutil
    import argparse
    from git import Repo, GitCommandError
    from clearml import Model
    from dotenv import load_dotenv

    """Commits and pushes changes to the remote repository."""
    commit_message = f"Update model weights: {model_id}"
    tag_name = f"{model_id}-{datetime.datetime.now().strftime('%Y%m%d')}"
    try:
        repo.index.commit(commit_message)
        repo.create_tag(tag_name, message="Model update")
        repo.git.push("origin", DEVELOPMENT_BRANCH)
        repo.git.push("origin", "--tags")
    except GitCommandError as e:
        print(f"Failed to commit and push changes: {e}")
        exit(1)


def cleanup_repo(repo_path):
    import os
    import datetime
    import shutil
    import argparse
    from git import Repo, GitCommandError
    from clearml import Model, Task
    from dotenv import load_dotenv

    """Safely remove the cloned repository directory."""
    shutil.rmtree(repo_path, ignore_errors=True)


def update_model(model_id, env_path, REPO_URL, DEVELOPMENT_BRANCH, project_name):
    import os
    import datetime
    import shutil
    import argparse
    from git import Repo, GitCommandError
    from clearml import Model, Task
    from dotenv import load_dotenv

    task = Task.init(
        project_name=project_name,
        task_name="Model Upload",
        task_type=Task.TaskTypes.custom,
    )
    """Fetches the trained model using its ID and updates it in the repository."""
    load_dotenv(dotenv_path=env_path)
    DEPLOY_KEY_PATH = os.getenv("DEPLOY_KEY_PATH")

    # Prepare repository and SSH key
    repo, repo_path = clone_repo(REPO_URL, DEVELOPMENT_BRANCH, DEPLOY_KEY_PATH)
    try:
        # Fetch the trained model
        model = Model(model_id=model_id)
        model_path = model.get_local_copy()

        # Update weights and push changes
        update_weights(repo, model_path)
        commit_and_push(repo, model_id, DEVELOPMENT_BRANCH)
    finally:
        cleanup_repo(repo_path)  # Ensure cleanup happens even if an error occurs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update model weights in GitHub repo using a ClearML model ID"
    )
    parser.add_argument(
        "--model_id",
        # required=True,
        help="The ClearML model ID to fetch and update",
        default="d2af953123b34ce7916de255cd793f92",
    )
    parser.add_argument(
        "--env_path",
        # required=True,
        help="Path to the .env file",
        default="/Users/apple/Desktop/AI_Studio/Introduction_to_MLOPS/First_Pipeline/.env",
    )
    parser.add_argument(
        "--repo_url",
        # required=True,
        help="Repository URL",
        default="git@github.com:GitarthVaishnav/Cifar10_SimpleFlaskApp.git",
    )
    parser.add_argument(
        "--development_branch",
        # required=True,
        help="Development branch name",
        default="development",
    )
    args = parser.parse_args()

    update_model(args.model_id, args.env_path, args.repo_url, args.development_branch)
