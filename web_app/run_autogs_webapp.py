#!/usr/bin/env python3
"""
Runner script for AutoG-S Web Application
"""

import os
import sys
import subprocess
import argparse
import socket
from pathlib import Path

def find_free_port(start_port=8501, max_attempts=10):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port + max_attempts}")

def main():
    """Run the AutoG-S web application"""
    parser = argparse.ArgumentParser(description="Run AutoG-S Web Application")
    parser.add_argument("--port", type=int, help="Port to run on (default: auto-detect starting from 8501)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--theme", type=str, choices=["light", "dark"], help="Streamlit theme")
    parser.add_argument("--no-auto-port", action="store_true", help="Don't auto-detect port, fail if specified port is busy")
    
    args = parser.parse_args()
    
    # Determine port to use
    if args.port:
        if args.no_auto_port:
            port = args.port
        else:
            # Check if specified port is available, otherwise find next available
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('', args.port))
                port = args.port
            except OSError:
                print(f"⚠️  Port {args.port} is busy, finding next available port...")
                port = find_free_port(args.port)
    else:
        # Auto-detect starting from 8501
        port = find_free_port(8501)
    
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
        print(f"   Access Key: {aws_access_key[:-4]}...")
        print(f"   Region: {region}")
    else:
        print("⚠️  AWS credentials not found in environment variables")
        print("   Please export them before running:")
        print("   export AWS_ACCESS_KEY_ID=<your_access_key>")
        print("   export AWS_SECRET_ACCESS_KEY=<your_secret_key>")
    
    # Construct streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(webapp_path),
        f"--server.port={port}",
        f"--server.address={args.host}",
        "--server.headless=true"
    ]
    
    if args.theme:
        cmd.append(f"--theme.base={args.theme}")
    
    try:
        print(f"🚀 Starting AutoG-S Web App on http://{args.host}:{port}")
        if port != args.port and args.port:
            print(f"   (Originally requested port {args.port} was busy)")
        print("   Press Ctrl+C to stop")
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\\n👋 Shutting down AutoG-S Web App...")
    except Exception as e:
        print(f"❌ Error running web app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()