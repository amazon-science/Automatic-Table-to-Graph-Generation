# -------------- install anaconda
bash
cd
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
# follow the installation prompts to install Anaconda
bash ~/Anaconda3-2024.10-1-Linux-x86_64.sh
source ~/.bashrc

# -------------- clone AutoG souce code
git clone https://github.com/amazon-science/Automatic-Table-to-Graph-Generation.git

# -------------- create a conda environment for the AutoG
cd Automatic-Table-to-Graph-Generation/

# for the mysqlclient linux requirement
sudo apt-get update
sudo apt-get install python3-dev default-libmysqlclient-dev build-essential pkg-config

bash multi-table-benchmark/conda/create_conda_env.sh -c -p 3.9 -t 2.1
# the env name will be autog-cpu
conda activate autog-cpu

# -------------- install other dependencies int autog-cpu environment
pip install codetiming humanfriendly sentence_transformers==3.3.0 nltk==3.9.1 torchdata==0.7 torchtext==0.16.0
pip install transformers==4.44.2
pip install llama-index llama-index-llms-bedrock
# for Graphviz
sudo apt-get install graphviz
# for nltk data
python -m nltk.downloader punkt_tab

# set PYTHONPATH according to your folder structure
export PYTHONPATH=/data/Automatic-Table-to-Graph-Generation/multi-table-benchmark

# -------------- clone deepjoin and setup git-lfs

# if the deepjoin folder is not exist
git clone https://github.com/mutong184/deepjoin

# setup git-lfs
sudo apt-get install git-lfs
git lfs install

# download large model files in deepjoin
cd deepjoin
git lfs pull


# =================== Using AutoG ================= #
# In this stage, we use MAG data as an example to test end2end pipeline
#   1. MAG data is downloeded from DGL and processed to get metadata
#   2. Need to manually call LLM API or Web console to get response
#   3. Output is a diagram, rather a text file.

# ---------- Download and process MAG data
# 1. modify the ./scripts/download.sh code
#    - specify the "dataset_path" if needed
#    - comment out commands for non-mag data except for mag

# cd to the root path of the repo
cd Automatic-Table-to-Graph-Generation
# if not in the autog-cpu conda env, activate it
conda activate autog-cpu
# download MAG dataset and process it to get the information.txt
bash scripts/download.sh

# ---------- run the AutoG
# 1. prepare the type.txt file
#   - copy the contents of identify in http://github.com/amazon-science/Automatic-Table-to-Graph-Generation/blob/main/prompts/identify.py
#   - replace the portion listed below with the contents of the information.txt under the mag/ folder.
#   - input the overall prompt+new contents into an LLM (e.g., Claud Sonet 3.5+) API or Web console to get answer.
#   - copy the answer (in RAW format!!!!!) and save to a file, named type.txt under the mag/ folder.

# 2. Option 1: run autog (manually call LLM), and explanation of these arguments
#   mag             -> name of the --dataset argument
#   /data/datasets  -> path of the folder that store data, .e.g, /data/datasets
#   autog-s         -> the method to run the model
#   type.txt        -> the name of file to save analysis results from LLM. Since we manually create this, this file is not used.
#   venue           -> Name of the task to fit the solution
python -m main.autog mag ./data/datasets/mag autog-s type.txt venue

# 2. Option 2: run autog2 (automatically interact with LLM for single or multiple rounds)
# Note: need to set up environment variables for AWS bedrock access before running the Python command.
export AWS_ACCESS_KEY_ID=<YOUR_AWS_ACCESS_KEY_ID>
export AWS_SECRET_ACCESS_KEY=<YOUR_AWS_SECRET_ACCESS_KEY>
export AWS_SESSION_TOKEN=<YOUR_AWS_SESSION_TOKEN>

python -m main.autog2 /data/datasets/mag/old sonnet4 autog-s mag:venue
