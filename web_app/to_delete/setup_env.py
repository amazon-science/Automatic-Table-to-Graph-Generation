"""
Environment setup for AutoG2 Web App
Handles PYTHONPATH and other environment configurations
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """Setup environment variables and paths for AutoG2 web app"""
    
    # Get the root directory (parent of web_app)
    current_dir = Path(__file__).parent
    root_dir = current_dir.parent
    
    # Add necessary paths to PYTHONPATH in the correct order
    paths_to_add = [
        str(current_dir),  # Add web_app directory FIRST for local imports
        str(root_dir),     # Then root directory for backend imports
        str(root_dir / "multi-table-benchmark"),
        str(root_dir / "dbinfer"),
        str(root_dir / "models"),
        str(root_dir / "prompts"),
        # Note: Don't add root_dir/utils to avoid conflicts with web_app/utils
    ]
    
    # Clear any existing paths that might cause conflicts
    paths_to_remove = [str(root_dir / "utils")]
    for path in paths_to_remove:
        if path in sys.path:
            sys.path.remove(path)
    
    # Add new paths
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # Set PYTHONPATH environment variable
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_paths = [p for p in paths_to_add if p not in current_pythonpath]
    
    if new_paths:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = ':'.join(new_paths) + ':' + current_pythonpath
        else:
            os.environ['PYTHONPATH'] = ':'.join(new_paths)
    
    print(f"✅ Environment setup complete")
    print(f"📁 Root directory: {root_dir}")
    print(f"📁 Web app directory: {current_dir}")
    print(f"🐍 Python paths added: {len(paths_to_add)}")
    
    # Debug: Print current working directory and first few paths
    print(f"🔍 Current working directory: {os.getcwd()}")
    print(f"🔍 First few sys.path entries:")
    for i, path in enumerate(sys.path[:5]):
        print(f"  {i}: {path}")
    
    return root_dir

if __name__ == "__main__":
    setup_environment()