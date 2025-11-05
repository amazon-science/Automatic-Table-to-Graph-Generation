#!/usr/bin/env python3
"""
Runner script for AutoG-S Web Application
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    """Run the AutoG-S web application"""
    parser = argparse.ArgumentParser(description="Run AutoG-S Web Application")
    parser.add_argument("--port", type=int, default=8501, help="Port to run on (default: 8501)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--theme", type=str, choices=["light", "dark"], help="Streamlit theme")
    
    args = parser.parse_args()
    
    # Get the web app file path
    current_dir = Path(__file__).parent
    webapp_path = current_dir / "AutoGS_WebApp.py"
    
    if not webapp_path.exists():
        print(f"Error: {webapp_path} not found!")
        sys.exit(1)
    
    # Check environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"🐍 Current conda environment: {conda_env}")
    
    if conda_env != 'autog-cpu':
        print("⚠️  Warning: You should be in the 'autog-cpu' conda environment")
        print("   Run: conda activate autog-cpu")
    
    # Check AWS credentials
    aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    
    if aws_access_key and aws_secret_key:
        print("✅ AWS credentials found in environment")
        region = os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
        print(f"   Access Key: {aws_access_key[:8]}...")
        print(f"   Region: {region}")
    else:
        print("⚠️  AWS credentials not found in environment variables")
        print("   Please export them before running:")
        print("   export AWS_ACCESS_KEY_ID=your_access_key")
        print("   export AWS_SECRET_ACCESS_KEY=your_secret_key")
    
    # Construct streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(webapp_path),
        f"--server.port={args.port}",
        f"--server.address={args.host}",
        "--server.headless=true"
    ]
    
    if args.theme:
        cmd.append(f"--theme.base={args.theme}")
    
    try:
        print(f"🚀 Starting AutoG-S Web App on http://{args.host}:{args.port}")
        print("   Press Ctrl+C to stop")
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\\n👋 Shutting down AutoG-S Web App...")
    except Exception as e:
        print(f"❌ Error running web app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()