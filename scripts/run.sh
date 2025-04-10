## run results for movielens


CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb MVLS ./newdatasets/movielens/autog/final ./newdatasets/movielens/autog/final r2n sage ratings -c configs/movielens/sage-r2ne.yaml

CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb MVLS ./newdatasets/movielens/expert ./newdatasets/movielens/expert r2n sage ratings -c configs/movielens/sage-r2ne.yaml


## mag venue

CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb MAG ./newdatasets/mag/autog/final ./newdatasets/mag/autog/final/ r2ne sage venue -c configs/mag/oracle-venue.yaml

CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb MAG ./newdatasets/mag/expert ./newdatasets/mag/expert/ r2ne sage venue -c configs/mag/oracle-venue.yaml

## mag cite (the performance may be lower, since it's a quick test, to get better performance please switch to better decoder and more early-stop epochs)

CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb MAG ./newdatasets/mag/autog/final ./newdatasets/mag/autog/final/ r2ne sage cite -c configs/mag/oracle-cite.yaml

CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb MAG ./newdatasets/mag/expert ./newdatasets/mag/expert/ r2ne sage cite -c configs/mag/oracle-cite.yaml

## mag year

## autog-s result, around ~50
CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb MAG ./newdatasets/mag/autog/final ./newdatasets/mag/autog/final/ r2n sage year -c configs/mag/oracle-year.yaml

## autog-a result 
CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb MAG ./newdatasets/mag/autog/round_0 ./newdatasets/mag/autog/round_0/ r2n sage year -c configs/mag/oracle-year.yaml

## expert result
CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb MAG ./newdatasets/mag/expert ./newdatasets/mag/expert/ r2n sage year -c configs/mag/oracle-year.yaml

# ## ieee-cis needs autog-a and pattern search to get not that bad results


CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb IEEE ./newdatasets/ieeecis/autog/expert ./newdatasets/ieeecis/expert/ r2n sage fraud -c configs/ieee-cis/sage.yaml

CUDA_VISIBLE_DEVICES=1 python3 -m main.rdb IEEE ./newdatasets/ieeecis/autog/round_0 ./newdatasets/ieeecis/autog/round_0 r2n sage fraud -c configs/ieee-cis/oracle-fraud.yaml


CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb RR ./newdatasets/retailrocket/autog/final ./newdatasets/retailrocket/autog/final/ r2n sage cvr -c configs/rr/oracle-cvr.yaml

CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb RR ./newdatasets/retailrocket/expert ./newdatasets/retailrocket/expert/ r2n sage cvr -c configs/rr/oracle-cvr.yaml

## avs dataset


## avs autog
CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb AVS ./newdatasets/avs/autog/final ./newdatasets/avs/autog/final/ r2n sage repeater -c configs/avs/oracle-repeater.yaml

## expert

CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb AVS ./newdatasets/avs/expert ./newdatasets/avs/expert/ r2n sage repeater -c configs/avs/oracle-repeater.yaml

## outbrain OBS

## autog
CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb OBS ./newdatasets/outbrain/autog/final ./newdatasets/outbrain/autog/final/ r2n sage ctr -c configs/outbrain-small/oracle-ctr.yaml

## expert 

CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb OBS ./newdatasets/outbrain/expert ./newdatasets/outbrain/expert/ r2n sage ctr -c configs/outbrain-small/oracle-ctr.yaml

## stackexchange

CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb STE ./newdatasets/stackexchange/expert ./newdatasets/stackexchange/expert/ r2n sage churn -c configs/stackexchange/sage.yaml

CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb STE ./newdatasets/stackexchange/expert ./newdatasets/stackexchange/expert/ r2n sage upvote -c configs/stackexchange/sage.yaml

CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb STE ./newdatasets/stackexchange/autog/final ./newdatasets/stackexchange/autog/final/ r2n sage churn -c configs/stackexchange/sage.yaml

CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb STE ./newdatasets/stackexchange/autog/final ./newdatasets/stackexchange/autog/final/ r2n sage upvote -c configs/stackexchange/sage.yaml

## diginetica

CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb DIG ./newdatasets/diginetica/autog/final/ ./newdatasets/diginetica/autog/final/ r2ne sage purchase -c multi-table-benchmark/hpo_results/diginetica/purchase/r2ne-sage.yaml

CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb DIG ./newdatasets/diginetica/expert/ ./newdatasets/diginetica/expert/ r2ne sage purchase -c multi-table-benchmark/hpo_results/diginetica/purchase/r2ne-sage.yaml


CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb DIG ./newdatasets/diginetica/autog/final/ ./newdatasets/diginetica/autog/final/ r2ne sage ctr -c multi-table-benchmark/hpo_results/diginetica/ctr/r2ne-sage.yaml

CUDA_VISIBLE_DEVICES=0 python3 -m main.rdb DIG ./newdatasets/diginetica/expert/ ./newdatasets/diginetica/expert/ r2ne sage ctr -c multi-table-benchmark/hpo_results/diginetica/ctr/r2ne-sage.yaml










