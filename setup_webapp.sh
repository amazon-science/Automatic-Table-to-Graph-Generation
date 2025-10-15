#!/bin/bash
# setup_myapp.sh
# Script to bootstrap the Streamlit + Bedrock + SQLite app structure

set -e

# Root folder
APP_NAME="myapp"
mkdir -p $APP_NAME

# Subfolders
mkdir -p $APP_NAME/data/uploads
mkdir -p $APP_NAME/data/results
mkdir -p $APP_NAME/db
mkdir -p $APP_NAME/services
mkdir -p $APP_NAME/ui
mkdir -p $APP_NAME/utils

# Empty files
touch $APP_NAME/app.py
touch $APP_NAME/requirements.txt
touch $APP_NAME/.env

touch $APP_NAME/db/models.py
touch $APP_NAME/db/database.py

touch $APP_NAME/services/profiling.py
touch $APP_NAME/services/llm_client.py
touch $APP_NAME/services/jobs.py

touch $APP_NAME/ui/forms.py
touch $APP_NAME/ui/results.py
touch $APP_NAME/ui/history.py

touch $APP_NAME/utils/config.py
touch $APP_NAME/utils/logger.py

echo "Project structure created under '$APP_NAME/'"
