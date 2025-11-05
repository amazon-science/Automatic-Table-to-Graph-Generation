# AWS Credentials Setup for AutoG-S Web App

## Overview
The AutoG-S web app requires AWS credentials to access AWS Bedrock for LLM processing. For security reasons, credentials should be set up in your terminal environment before running the web app.

## Setup Methods

### Method 1: Using AWS CLI (Recommended)

1. **Install AWS CLI** (if not already installed):
   ```bash
   # Ubuntu/Debian
   sudo apt-get install awscli
   
   # macOS
   brew install awscli
   
   # Or follow: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
   ```

2. **Configure AWS CLI**:
   ```bash
   aws configure
   ```
   
   You'll be prompted to enter:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region name (e.g., `us-west-2`)
   - Default output format (e.g., `json`)

3. **Test your credentials**:
   ```bash
   aws sts get-caller-identity
   ```

### Method 2: Environment Variables

Set the following environment variables in your terminal:

```bash
export AWS_ACCESS_KEY_ID=your_access_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_key_here
export AWS_SESSION_TOKEN=your_session_token_here  # Optional for temporary credentials
export AWS_DEFAULT_REGION=us-west-2
```

### Method 3: Using the Setup Script

Run the provided setup script:

```bash
cd web_app
chmod +x setup_aws_credentials.sh
./setup_aws_credentials.sh
```

## Required Permissions

Your AWS credentials need the following permissions for AutoG-S:

- `bedrock:InvokeModel` - To call LLM models
- Access to specific Bedrock models:
  - `anthropic.claude-3-5-sonnet-20241022-v2:0`
  - `anthropic.claude-3-sonnet-20240229-v1:0`
  - `anthropic.claude-3-haiku-20240307-v1:0`

## Running the Web App

Once your AWS credentials are configured:

```bash
cd web_app
python run_autogs_webapp.py
```

The web app will automatically detect and validate your AWS credentials.

## Troubleshooting

### Credentials Not Found
If you see "AWS credentials not found", make sure you've set them in the same terminal session where you're running the web app.

### Invalid Credentials
If credentials are invalid:
1. Double-check your Access Key ID and Secret Access Key
2. Verify the credentials have the necessary permissions
3. For temporary credentials, ensure the session token is still valid

### Region Issues
Make sure your AWS region supports Bedrock and the models you want to use. Recommended regions:
- `us-west-2` (Oregon)
- `us-east-1` (N. Virginia)

## Security Best Practices

- ✅ **DO**: Set credentials via environment variables or AWS CLI
- ✅ **DO**: Use IAM roles with minimal required permissions
- ✅ **DO**: Rotate credentials regularly
- ❌ **DON'T**: Hardcode credentials in code
- ❌ **DON'T**: Share credentials in chat or email
- ❌ **DON'T**: Commit credentials to version control

## Getting AWS Credentials

If you don't have AWS credentials:

1. **Sign up for AWS**: https://aws.amazon.com/
2. **Create IAM user**: Go to IAM console → Users → Create user
3. **Attach policies**: Add `AmazonBedrockFullAccess` or create custom policy
4. **Generate access keys**: Security credentials → Create access key

For more information: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html