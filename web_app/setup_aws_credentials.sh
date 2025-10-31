#!/bin/bash

# AutoG2 AWS Credentials Setup Script
# This script helps you set up AWS credentials for the AutoG2 web app

echo "🔗 AutoG2 AWS Credentials Setup"
echo "================================"
echo

# Check if AWS CLI is installed
if command -v aws &> /dev/null; then
    echo "✅ AWS CLI is installed"
    
    # Check if AWS CLI is configured
    if aws configure list &> /dev/null; then
        echo "✅ AWS CLI appears to be configured"
        echo
        echo "Current AWS configuration:"
        aws configure list
        echo
        
        read -p "Do you want to reconfigure AWS CLI? (y/N): " reconfigure
        if [[ $reconfigure =~ ^[Yy]$ ]]; then
            echo "Running 'aws configure'..."
            aws configure
        fi
    else
        echo "⚠️  AWS CLI is not configured"
        echo "Running 'aws configure' to set up credentials..."
        aws configure
    fi
else
    echo "❌ AWS CLI is not installed"
    echo
    echo "Please install AWS CLI first:"
    echo "  - Ubuntu/Debian: sudo apt-get install awscli"
    echo "  - macOS: brew install awscli"
    echo "  - Or follow: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    echo
    echo "Alternatively, you can set environment variables manually:"
    echo
    echo "export AWS_ACCESS_KEY_ID=your_access_key_here"
    echo "export AWS_SECRET_ACCESS_KEY=your_secret_key_here"
    echo "export AWS_SESSION_TOKEN=your_session_token_here  # Optional for temporary credentials"
    echo "export AWS_DEFAULT_REGION=us-west-2"
    echo
    exit 1
fi

echo
echo "🧪 Testing AWS credentials..."

# Test AWS credentials
if aws sts get-caller-identity &> /dev/null; then
    echo "✅ AWS credentials are working!"
    echo
    echo "Account information:"
    aws sts get-caller-identity --output table
    echo
    echo "🚀 You can now run the AutoG2 web app:"
    echo "   cd web_app"
    echo "   python run_autog2_webapp.py"
else
    echo "❌ AWS credentials test failed"
    echo
    echo "Please check your credentials and try again."
    echo "You may need to:"
    echo "  1. Verify your Access Key ID and Secret Access Key"
    echo "  2. Check if your credentials have the necessary permissions"
    echo "  3. Ensure your session token is valid (if using temporary credentials)"
    echo
    echo "For AWS Bedrock access, make sure your credentials have:"
    echo "  - bedrock:InvokeModel permission"
    echo "  - Access to the specific models you want to use"
fi

echo
echo "📚 For more information about AWS credentials:"
echo "   https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html"