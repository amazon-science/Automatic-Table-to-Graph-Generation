dataset_path="/workspace/datasets"

export LD_LIBRARY_PATH="/opt/conda/envs/dbinfer-gpu/lib:$LD_LIBRARY_PATH"

mkdir -p $dataset_path

## download movielens

mkdir -p "$dataset_path/movielens/raw"
curl -C - -o "$dataset_path/movielens/raw/ml-latest-small.zip" https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip -o "$dataset_path/movielens/raw/ml-latest-small.zip" -d "$dataset_path/movielens/raw"
mkdir -p "$dataset_path/movielens/old"
mkdir -p "$dataset_path/movielens/expert"
mkdir -p "$dataset_path/movielens/old/data"
mkdir -p "$dataset_path/movielens/expert/data"
mkdir -p "$dataset_path/movielens/old/ratings"
mkdir -p "$dataset_path/movielens/expert/ratings"

python3 -m main.preprocessing_dataset mvls $dataset_path

