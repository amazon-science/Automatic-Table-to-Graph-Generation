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

try:
    import tomli as tomllib  # Python < 3.11
except ImportError:
    try:
        import tomllib  # Python >= 3.11
    except ImportError:
        import toml as tomllib  # Fallback to toml package

def load_streamlit_config(config_path):
    """Load Streamlit config.toml file"""
    try:
        if hasattr(tomllib, 'load'):
            # tomllib (Python 3.11+) requires binary mode
            with open(config_path, 'rb') as f:
                return tomllib.load(f)
        else:
            # toml package uses text mode
            with open(config_path, 'r') as f:
                return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"⚠️  Warning: Could not load config.toml: {e}")
        return {}

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
    parser.add_argument("--use-instance-role", action="store_true", help="Use EC2 instance role for AWS credentials (instead of environment variables)")
    
    args = parser.parse_args()
    
    # Get the web app file path
    current_dir = Path(__file__).parent
    webapp_path = current_dir / "AutoGS_WebApp.py"
    config_path = current_dir / ".streamlit" / "config.toml"
    
    # Load config.toml
    config = load_streamlit_config(config_path)
    server_config = config.get('server', {})
    
    # Determine port to use (CLI > config.toml > default)
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
        # Use port from config.toml if available
        port = server_config.get('port')
    
    # Determine host (CLI > config.toml > default)
    if args.host != "0.0.0.0":
        host = args.host
    else:
        host = server_config.get('address', '0.0.0.0')
    
    # Determine headless mode (config.toml > default)
    headless = server_config.get('headless', True)
    
    # Check environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"🐍 Current conda environment: {conda_env}")
    
    if conda_env != 'autog-cpu':
        print("⚠️  Warning: You should be in the 'autog-cpu' conda environment")
        print("   Run: conda activate autog-cpu")
    
    # Setup AWS credentials (environment variables or instance role)
    # Set USE_INSTANCE_ROLE env var so AutoGS_WebApp.py uses the same setting
    if args.use_instance_role:
        os.environ['USE_INSTANCE_ROLE'] = 'true'
    
    sys.path.insert(0, str(current_dir))  # Add web_app to path for imports
    try:
        from webutils.aws_credentials import setup_aws_credentials
        if setup_aws_credentials(use_instance_role=args.use_instance_role):
            print("✅ AWS credentials configured")
        else:
            if args.use_instance_role:
                print("⚠️  No AWS credentials found from EC2 instance role")
                print("   Make sure you're running on an EC2 instance with an IAM role attached")
            else:
                print("⚠️  No AWS credentials found in environment variables")
                print("   Tip: Use --use-instance-role to use EC2 instance role instead")
                print("   Or export environment variables:")
                print("   export AWS_ACCESS_KEY_ID=<your_access_key>")
                print("   export AWS_SECRET_ACCESS_KEY=<your_secret_key>")
    except ImportError as e:
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        if aws_access_key and aws_secret_key:
            print("✅ AWS credentials found in environment")
            region = os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
            print(f"   Access Key: {aws_access_key[:-4]}...")
            print(f"   Region: {region}")
        else:
            print("⚠️  No AWS credentials found")
            print("   Please export them before running:")
            print("   export AWS_ACCESS_KEY_ID=<your_access_key>")
            print("   export AWS_SECRET_ACCESS_KEY=<your_secret_key>")
    
    # Construct streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(webapp_path),
    ]
    
    # Add server configuration (only if different from config.toml or explicitly set)
    if port is not None:
        cmd.append(f"--server.port={port}")
    if host != "0.0.0.0" or args.host != "0.0.0.0":
        cmd.append(f"--server.address={host}")
    if headless:
        cmd.append("--server.headless=true")
    
    if args.theme:
        cmd.append(f"--theme.base={args.theme}")
    
    try:
        # Display startup info
        if port is not None:
            print(f"🚀 Starting AutoG-S Web App on http://{host}:{port}")
            if port != args.port and args.port:
                print(f"   (Originally requested port {args.port} was busy)")
            if not args.port and server_config.get('port'):
                print(f"   Using port from config.toml")
        else:
            print(f"🚀 Starting AutoG-S Web App on http://{host}:<default port>")
            print("   Using default Streamlit port (8501)")
        
        # Show config source
        config_sources = []
        if server_config:
            config_sources.append("config.toml")
        if args.port or args.host != "0.0.0.0" or args.theme:
            config_sources.append("CLI args")
        if config_sources:
            print(f"   Config from: {', '.join(config_sources)}")
        
        print("   Press Ctrl+C to stop")
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\\n👋 Shutting down AutoG-S Web App...")
    except Exception as e:
        print(f"❌ Error running web app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()