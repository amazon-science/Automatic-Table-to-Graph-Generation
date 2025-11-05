#!/usr/bin/env python3
"""
Startup script for AutoG2 Web App
Handles environment setup and launches Streamlit
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main function to setup and run the web app"""
    
    # Setup environment
    from setup_env import setup_environment
    root_dir = setup_environment()
    
    # Check if we're in the correct conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"🐍 Current conda environment: {conda_env}")
    
    if conda_env != 'autog-cpu':
        print("⚠️  Warning: You should be in the 'autog-cpu' conda environment")
        print("   Run: conda activate autog-cpu")
    
    # Check for required AWS credentials
    aws_keys = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    missing_keys = [key for key in aws_keys if not os.environ.get(key)]
    
    if missing_keys:
        print("⚠️  AWS credentials not found in environment variables")
        print("   You can set them in the web app interface")
    else:
        print("✅ AWS credentials found in environment")
    
    # Check if deepjoin path exists
    deepjoin_path = root_dir / "deepjoin"
    if not deepjoin_path.exists():
        print("⚠️  DeepJoin not found. Some features may not work.")
        print(f"   Expected path: {deepjoin_path}")
    else:
        print("✅ DeepJoin found")
    
    # Check for graphviz (required for schema generation)
    try:
        import subprocess
        result = subprocess.run(['dot', '-V'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Graphviz found (schema generation available)")
        else:
            print("⚠️  Graphviz not found. Schema diagrams may not generate.")
            print("   Install with: sudo apt-get install graphviz")
    except FileNotFoundError:
        print("⚠️  Graphviz not found. Schema diagrams may not generate.")
        print("   Install with: sudo apt-get install graphviz")
    
    # Launch Streamlit
    print("\n🚀 Starting AutoG2 Web App...")
    print("   Open your browser to: http://localhost:8501")
    
    # Change to web_app directory
    os.chdir(Path(__file__).parent)
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down AutoG2 Web App...")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())