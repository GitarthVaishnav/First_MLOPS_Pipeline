from clearml import Model
from git import Repo, GitCommandError
import os
import datetime
import shutil
from dotenv import load_dotenv  # Import load_dotenv from python-dotenv

# Specify the path to your .env file if it's not in the default location
# For example: /path/to/your/.env
env_path = "../.env"
load_dotenv(dotenv_path=env_path)

# Now, you can safely load DEPLOY_KEY_PATH from your .env file
DEPLOY_KEY_PATH = os.getenv("DEPLOY_KEY_PATH")
REPO_URL = "git@github.com:GitarthVaishnav/Cifar10_SimpleFlaskApp.git"
DEVELOPMENT_BRANCH = "development"


def configure_ssh_key():
    from clearml import Model
    from git import Repo, GitCommandError
    import os
    import datetime
    import shutil
    from dotenv import load_dotenv  # Import load_dotenv from python-dotenv

    """Configure Git to use the SSH deploy key for operations."""
    os.environ["GIT_SSH_COMMAND"] = f"ssh -i {DEPLOY_KEY_PATH} -o IdentitiesOnly=yes"


def clone_repo(branch="development"):
    from clearml import Model
    from git import Repo, GitCommandError
    import os
    import datetime
    import shutil
    from dotenv import load_dotenv  # Import load_dotenv from python-dotenv

    repo_path = REPO_URL.split("/")[-1].split(".git")[0]
    try:
        repo = Repo.clone_from(REPO_URL, repo_path, branch=branch, single_branch=True)
        return (
            repo,
            repo_path,
        )  # Return both the Repo object and the path for later cleanup
    except GitCommandError as e:
        print(f"Failed to clone repository: {e}")
        exit(1)


def ensure_archive_dir(repo):
    from clearml import Model
    from git import Repo, GitCommandError
    import os
    import datetime
    import shutil
    from dotenv import load_dotenv  # Import load_dotenv from python-dotenv

    """Ensures the archive directory exists within weights."""
    archive_path = os.path.join(repo.working_tree_dir, "weights", "archive")
    os.makedirs(archive_path, exist_ok=True)


def archive_existing_model(repo):
    from clearml import Model
    from git import Repo, GitCommandError
    import os
    import datetime
    import shutil
    from dotenv import load_dotenv  # Import load_dotenv from python-dotenv

    """Archives existing model weights."""
    weights_path = os.path.join(repo.working_tree_dir, "weights")
    model_file = os.path.join(weights_path, "model.h5")
    if os.path.exists(model_file):
        today = datetime.date.today().strftime("%Y%m%d")
        archived_model_file = os.path.join(weights_path, "archive", f"model-{today}.h5")
        os.rename(model_file, archived_model_file)


def update_weights(repo, model_path):
    from clearml import Model
    from git import Repo, GitCommandError
    import os
    import datetime
    import shutil
    from dotenv import load_dotenv  # Import load_dotenv from python-dotenv

    """Updates the model weights in the repository."""
    weights_path = os.path.join(repo.working_tree_dir, "weights")
    ensure_archive_dir(repo)
    archive_existing_model(repo)
    target_model_path = os.path.join(weights_path, "model.h5")
    os.rename(model_path, target_model_path)
    repo.index.add([target_model_path])


def commit_and_push(repo, model_id):
    from clearml import Model
    from git import Repo, GitCommandError
    import os
    import datetime
    import shutil
    from dotenv import load_dotenv  # Import load_dotenv from python-dotenv

    """Commits and pushes changes to the remote repository."""
    commit_message = f"Update model weights: {model_id}"
    tag_name = f"{model_id}-{datetime.datetime.now().strftime('%Y%m%d')}"
    try:
        repo.index.commit(commit_message)
        repo.create_tag(tag_name, message="Model update")
        repo.git.push("origin", DEVELOPMENT_BRANCH)
        repo.git.push("origin", tag_name)
    except GitCommandError as e:
        print(f"Failed to commit and push changes: {e}")
        exit(1)


def cleanup_repo(repo_path):
    from clearml import Model
    from git import Repo, GitCommandError
    import os
    import datetime
    import shutil
    from dotenv import load_dotenv  # Import load_dotenv from python-dotenv

    """Safely remove the cloned repository directory."""
    shutil.rmtree(repo_path, ignore_errors=True)


def update_model(model_id):
    from clearml import Model
    from git import Repo, GitCommandError
    import os
    import datetime
    import shutil
    from dotenv import load_dotenv  # Import load_dotenv from python-dotenv

    """Fetches the trained model using its ID and updates it in the repository."""
    # Fetch the trained model
    model = Model(model_id=model_id)
    model_path = model.get_local_copy()

    # Prepare repository and SSH key
    configure_ssh_key()
    repo, repo_path = clone_repo(DEVELOPMENT_BRANCH)
    try:
        # Update weights and push changes
        update_weights(repo, model_path)
        commit_and_push(repo, model_id)
    finally:
        cleanup_repo(repo_path)  # Ensure cleanup happens even if an error occurs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Update model weights in GitHub repo using a ClearML model ID"
    )
    parser.add_argument(
        "--model_id",
        required=True,
        help="The ClearML model ID to fetch and update",
        default="fc6c35419c97405495f3b1713446ca58",
    )
    args = parser.parse_args()

    update_model(args.model_id)
