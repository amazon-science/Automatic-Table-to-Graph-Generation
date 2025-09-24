## IEEE-CIS RR Movielens Outbrain MAG AVS diginetica ESCI Stackexchange

dataset_path="/data/datasets"

mkdir -p $dataset_path

## download movielens

# mkdir -p "$dataset_path/movielens/raw"
# curl -C - -o "$dataset_path/movielens/raw/ml-latest-small.zip" https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
# unzip -o "$dataset_path/movielens/raw/ml-latest-small.zip" -d "$dataset_path/movielens/raw"
# mkdir -p "$dataset_path/movielens/old"
# mkdir -p "$dataset_path/movielens/expert"
# mkdir -p "$dataset_path/movielens/old/data"
# mkdir -p "$dataset_path/movielens/expert/data"
# mkdir -p "$dataset_path/movielens/old/ratings"
# mkdir -p "$dataset_path/movielens/expert/ratings"

# python3 -m main.preprocessing_dataset mvls $dataset_path

# generate folder for autog  


# download ieeecis

# mkdir -p "$dataset_path/ieeecis/raw"

# kaggle competitions download -c ieee-fraud-detection --path "$dataset_path/ieeecis/raw"
# unzip -o "$dataset_path/ieeecis/raw/ieee-fraud-detection.zip" -d "$dataset_path/ieeecis/raw"

# mkdir -p "$dataset_path/ieeecis/old"
# mkdir -p "$dataset_path/ieeecis/expert"
# mkdir -p "$dataset_path/ieeecis/old/data"
# mkdir -p "$dataset_path/ieeecis/expert/data"
# mkdir -p "$dataset_path/ieeecis/old/fraud"
# mkdir -p "$dataset_path/ieeecis/expert/fraud"

# python3 -m main.preprocessing_dataset IEEE-CIS $dataset_path

# download MAG


mkdir -p "$dataset_path/mag/raw"

mkdir -p "$dataset_path/mag/old"
mkdir -p "$dataset_path/mag/expert"
mkdir -p "$dataset_path/mag/old/data"
mkdir -p "$dataset_path/mag/expert/data"
mkdir -p "$dataset_path/mag/old/venue"
mkdir -p "$dataset_path/mag/expert/venue"
mkdir -p "$dataset_path/mag/old/cite"
mkdir -p "$dataset_path/mag/expert/cite"
mkdir -p "$dataset_path/mag/old/year"
mkdir -p "$dataset_path/mag/expert/year"

DBB_DATASET_HOME="$dataset_path/mag/raw" python3 -m dbinfer.main download mag

python3 -m main.preprocessing_dataset MAG $dataset_path

# download outbrain
# mkdir -p "$dataset_path/outbrain/raw"
# mkdir -p "$dataset_path/outbrain/old"
# mkdir -p "$dataset_path/outbrain/expert"
# mkdir -p "$dataset_path/outbrain/old/data"
# mkdir -p "$dataset_path/outbrain/expert/data"
# mkdir -p "$dataset_path/outbrain/old/ctr"
# mkdir -p "$dataset_path/outbrain/expert/ctr"

# DBB_DATASET_HOME="$dataset_path/outbrain/raw" python3 -m dbinfer.main download outbrain-small

# python3 -m main.preprocessing_dataset outbrain


# download avs

# mkdir -p "$dataset_path/avs/raw"
# mkdir -p "$dataset_path/avs/old"
# mkdir -p "$dataset_path/avs/expert"
# mkdir -p "$dataset_path/avs/old/data"
# mkdir -p "$dataset_path/avs/expert/data"
# mkdir -p "$dataset_path/avs/old/repeater"
# mkdir -p "$dataset_path/avs/expert/repeater"

# DBB_DATASET_HOME="$dataset_path/avs/raw" python3 -m dbinfer.main download avs

# python3 -m main.preprocessing_dataset avs

# download retailrocket

# mkdir -p "$dataset_path/retailrocket/raw"
# mkdir -p "$dataset_path/retailrocket/old"
# mkdir -p "$dataset_path/retailrocket/expert"
# mkdir -p "$dataset_path/retailrocket/old/data"
# mkdir -p "$dataset_path/retailrocket/expert/data"
# mkdir -p "$dataset_path/retailrocket/old/cvr"
# mkdir -p "$dataset_path/retailrocket/expert/cvr"

# mkdir -p "$dataset_path/retailrocket/realold/"
# mkdir -p "$dataset_path/retailrocket/realold/data"
# mkdir -p "$dataset_path/retailrocket/realold/cvr"

# DBB_DATASET_HOME="$dataset_path/retailrocket/raw" python3 -m dbinfer.main download retailrocket

# python3 -m main.preprocessing_dataset RR

# download diginetica

## we find this dataset is problematic
# mkdir -p "$dataset_path/diginetica/raw"
# mkdir -p "$dataset_path/diginetica/old"
# mkdir -p "$dataset_path/diginetica/expert"
# mkdir -p "$dataset_path/diginetica/old/data"
# mkdir -p "$dataset_path/diginetica/expert/data"
# mkdir -p "$dataset_path/diginetica/old/ctr"
# mkdir -p "$dataset_path/diginetica/expert/ctr"
# mkdir -p "$dataset_path/diginetica/old/purchase"
# mkdir -p "$dataset_path/diginetica/expert/purchase"

# DBB_DATASET_HOME="$dataset_path/diginetica/raw" python3 -m dbinfer.main download diginetica

# python3 -m main.preprocessing_dataset diginetica

# download stackexchange

# mkdir -p "$dataset_path/stackexchange/raw"
# mkdir -p "$dataset_path/stackexchange/old"
# mkdir -p "$dataset_path/stackexchange/expert"
# mkdir -p "$dataset_path/stackexchange/old/data"
# mkdir -p "$dataset_path/stackexchange/expert/data"
# mkdir -p "$dataset_path/stackexchange/old/churn"
# mkdir -p "$dataset_path/stackexchange/expert/churn"
# mkdir -p "$dataset_path/stackexchange/old/upvote"
# mkdir -p "$dataset_path/stackexchange/expert/upvote"

# DBB_DATASET_HOME="$dataset_path/stackexchange/raw" python3 -m dbinfer.main download stackexchange

# python3 -m main.preprocessing_dataset stackexchange

# git clone https://github.com/mutong184/deepjoin.git