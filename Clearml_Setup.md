
# Getting Started with ClearML

ClearML is an essential tool for managing Machine Learning experiments, allowing users to track, organize, and optimize your projects with ease. This guide is designed to help you get started by signing up for ClearML's free web instance, installing ClearML in your coding environment, and initializing it with your configuration.

## Step 1: Sign Up for the Free ClearML Web Instance

1. Visit the ClearML Web Instance [sign-up page](https://app.clear.ml/login).
2. Complete the registration form with your information using UTS Credentials where possible.
3. Confirm your email address by clicking on the link sent to your inbox.
4. Log in to your ClearML account.

## Step 2: Get Your ClearML Configuration

To connect your coding environment with ClearML, you'll need your API credentials.

1. Go to your profile settings in the ClearML web interface.
2. Find the API credentials section and generate a new set or use existing ones.
3. Copy your API key and secret component, which looks like the following.
    ```bash
    api { 
        # Gitarth Vaishnav's workspace (url depends on where it is hosted)
        web_server: https://app.clear.ml
        api_server: https://api.clear.ml
        files_server: https://files.clear.ml
        # Some Credentials
        credentials {
            "access_key" = "SomeAccessKey"
            "secret_key"  = "SomeSecretKey"
        }
    }
    ```

## Step 3: Install ClearML in Your Coding Environment

### For Google Colab

1. In a new Google Colab notebook, run: `!pip install clearml`.

### For Local Environments and AWS SageMaker

Open a terminal on your machine or in your AWS SageMaker instance and follow these steps:

1. **For Local Environments:**
   - Ensure Python is installed on your machine.
   - Run `pip install clearml` to install ClearML.

2. **For AWS SageMaker:**
   - In your SageMaker notebook, select the terminal option or open a terminal session.
   - Execute `pip install clearml` within the terminal to install ClearML.

## Step 4: Initialize ClearML

With ClearML installed, the next step is to configure it with your API credentials.

1. In your terminal or notebook, run `clearml-init`.
2. When prompted, enter your API credentials.
3. Alternatively, for some notebook environments, you can also configure ClearML programmatically:
   ```python
   from clearml import Task
   Task.set_credentials(api_host='https://app.clear.ml', 
   web_host='https://app.clear.ml', 
   files_host='https://app.clear.ml', 
   key='YOUR_KEY', 
   secret='YOUR_SECRET')
   ```

You're now ready to leverage ClearML for tracking your experiments, models, and datasets.

## Additional Resources

- **ClearML Documentation**: For detailed guides, tutorials, and API references, visit the [official documentation](https://clear.ml/docs/latest/).
- **YouTube Tutorials**: visit the [official playlist](https://youtube.com/playlist?list=PLMdIlCuMqSTnoC45ME5_JnsJX0zWqDdlO&si=Nw9_7X63v-0Vvh4Y).
- **ClearML GitHub**: Explore the [ClearML GitHub repository](https://github.com/allegroai/clearml) for examples and in-depth guides.

Exploring ClearML's documentation and resources is a great way to familiarize yourself with its full range of features and capabilities.