# Model_list: MM_SAGE, MMD_GCN, MMD_SAGE, GraphSAGE, GCN, GAT, APPNP, GCN2-ogb, GCN2, MixHop, GPRGNN, H2GCN
# Dataset_list: Cora, CiteSeer, PubMed, Reddit, BlogCatalog, 
#               Cornell, Texas, Wisconsin, Actor, 'Amherst41', 'Hamilton46', 'Penn94', "chameleon", "squirrel"
#               'Georgetown15'


# Cornell
# thres_prunning=0.6 for GPRGNN, GAT
# thres_prunning=0.3 for GCN2
python3 main_ae.py \
    --model GCN \
    --num_layer 2 \
    --repeat 3 \
    --num_epoch 200 \
    --lr 0.01 \
    --wd 5e-3 \
    --gpu 0 \
    --data_dir ../data \
    --dataset Cornell \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --seed 0 \
    --graph_learn \
    --lr_gl 0.001 \
    --wd_gl 5e-3 \
    --thres_min_deg 3 \
    --thres_min_deg_ratio 1.0 \
    --window [10000,10000] \
    --shuffle [False,False] \
    --drop_last [False,False] \
    --epoch_train_gl 200 \
    --epoch_finetune_gl 30 \
    --seed_gl 0 \
    --k 8 \
    --cat_self True \
    --prunning True \
    --thres_prunning 0. \
    

# Texas
python3 main_ae.py \
    --model GCN \
    --num_layer 2 \
    --repeat 3 \
    --num_epoch 200 \
    --lr 0.01 \
    --wd 5e-3 \
    --gpu 0 \
    --data_dir ../data \
    --dataset Texas \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --seed 0 \
    --graph_learn \
    --lr_gl 0.001 \
    --wd_gl 5e-3 \
    --thres_min_deg 3 \
    --thres_min_deg_ratio 1.0 \
    --window [10000,10000] \
    --shuffle [False,False] \
    --drop_last [False,False] \
    --epoch_train_gl 200 \
    --epoch_finetune_gl 30 \
    --seed_gl 0 \
    --k 8 \
    --cat_self True \
    --prunning True \
    --thres_prunning 0.6 \

# Wisconsin
python3 main_ae.py \
    --model GCN \
    --num_layer 2 \
    --repeat 3 \
    --num_epoch 200 \
    --lr 0.01 \
    --wd 5e-3 \
    --gpu 1 \
    --data_dir ../data \
    --dataset Wisconsin \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --seed 0 \
    --graph_learn \
    --lr_gl 0.001 \
    --wd_gl 5e-3 \
    --thres_min_deg 3 \
    --thres_min_deg_ratio 1.0 \
    --window [10000,10000] \
    --shuffle [False,False] \
    --drop_last [False,False] \
    --epoch_train_gl 200 \
    --epoch_finetune_gl 30 \
    --seed_gl 0 \
    --k 8 \
    --cat_self True \
    --prunning True \
    --thres_prunning 0.6 \


# chamelon
python3 main_ae.py \
    --model GCN \
    --num_layer 2 \
    --repeat 3 \
    --num_epoch 200 \
    --lr 0.01 \
    --wd 5e-3 \
    --gpu 0 \
    --data_dir ../data \
    --dataset chameleon \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --seed 0 \
    --graph_learn \
    --lr_gl 0.001 \
    --wd_gl 5e-3 \
    --thres_min_deg 3 \
    --thres_min_deg_ratio 1.0 \
    --window [10000,10000] \
    --shuffle [False,False] \
    --drop_last [False,False] \
    --epoch_train_gl 200 \
    --epoch_finetune_gl 30 \
    --seed_gl 0 \
    --k 8 \
    --cat_self False \
    --prunning True \
    --thres_prunning 0.6 \
    --use_cpu_cache


# squirrel
python3 main_ae.py \
    --model GAT \
    --num_layer 2 \
    --repeat 3 \
    --num_epoch 200 \
    --lr 0.01 \
    --wd 5e-3 \
    --gpu 2 \
    --data_dir ../data \
    --dataset squirrel \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --seed 0 \
    --graph_learn \
    --lr_gl 0.001 \
    --wd_gl 5e-3 \
    --thres_min_deg 3 \
    --thres_min_deg_ratio 1.0 \
    --window [10000,10000] \
    --shuffle [False,False] \
    --drop_last [False,False] \
    --epoch_train_gl 200 \
    --epoch_finetune_gl 30 \
    --seed_gl 0 \
    --k 8 \
    --cat_self False \
    --prunning True \
    --thres_prunning 0.6 \

# Actor
python3 main_ae.py \
    --model GCN \
    --num_layer 2 \
    --repeat 3 \
    --num_epoch 200 \
    --lr 0.01 \
    --wd 5e-3 \
    --gpu 0 \
    --data_dir ../data \
    --dataset Actor \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --seed 0 \
    --graph_learn \
    --lr_gl 0.001 \
    --wd_gl 5e-3 \
    --thres_min_deg 10 \
    --thres_min_deg_ratio 1.0 \
    --window [5000,5000] \
    --shuffle [False,False] \
    --drop_last [False,False] \
    --epoch_train_gl 200 \
    --epoch_finetune_gl 30 \
    --seed_gl 0 \
    --k 8 \
    --cat_self False \
    --prunning True \
    --thres_prunning 0.5 \


# fb100(Penn94) # baseline
# wd=1e-4 for GCN2, 1e-3 for others
# num_layer = 2/32/64 for GCN2
python3 main_ae.py \
    --model GCN2 \
    --num_layer 64 \
    --repeat 3 \
    --num_epoch 300 \
    --lr 0.01 \
    --wd 1e-4 \
    --gpu 2 \
    --data_dir ../data \
    --dataset fb100 \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --seed 0 \


# fb100(Penn94) # for GraphSAGE,H2GCN
num_layer = 2 for GCN2(also try 32, 64)
python3 main_ae.py \
    --model GPRGNN \
    --num_layer 2 \
    --repeat 3 \
    --num_epoch 300 \
    --lr 0.01 \
    --wd 1e-4 \
    --gpu 0 \
    --data_dir ../data \
    --dataset fb100 \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --seed 0 \
    --graph_learn \
    --lr_gl 0.01 \
    --wd_gl 1e-4 \
    --thres_min_deg 50 \
    --thres_min_deg_ratio 1.0 \
    --window [10000,10000] \
    --shuffle [True,True] \
    --drop_last [True,True] \
    --epoch_train_gl 10 \
    --epoch_finetune_gl 3 \
    --seed_gl 0 \
    --k 8 \
    --cat_self True \
    --prunning False \
    --thres_prunning -1.0 \
    --use_cpu_cache \

# fb100(Penn94) GCN,GAT,GCNII,APPNP
python3 main_ae.py \
    --model APPNP \
    --num_layer 2 \
    --repeat 3 \
    --num_epoch 200 \
    --lr 0.01 \
    --wd 5e-3 \
    --gpu 2 \
    --data_dir ../data \
    --dataset fb100 \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --seed 0 \
    --graph_learn \
    --lr_gl 0.001 \
    --wd_gl 5e-3 \
    --thres_min_deg 50 \
    --thres_min_deg_ratio 1.0 \
    --window [10000,10000] \
    --shuffle [True,True] \
    --drop_last [True,True] \
    --epoch_train_gl 10 \
    --epoch_finetune_gl 3 \
    --seed_gl 0 \
    --k 8 \
    --cat_self True \
    --prunning True \
    --thres_prunning 0 \
    --use_cpu_cache \



# Flickr, GCN, GAT
python3 main_ae.py \
    --model APPNP \
    --num_layer 2 \
    --repeat 3 \
    --num_epoch 200 \
    --lr 0.01 \
    --wd 5e-3 \
    --gpu 3 \
    --data_dir ../data \
    --dataset Flickr \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --graph_learn \
    --lr_gl 0.001 \
    --wd_gl 5e-3 \
    --thres_min_deg 10 \
    --thres_min_deg_ratio 1.0 \
    --window [10000,10000] \
    --shuffle [True,True] \
    --drop_last [True,True] \
    --epoch_train_gl 10 \
    --epoch_finetune_gl 3 \
    --k 6 \
    --cat_self False \
    --prunning False \
    --thres_prunning 0. \
    --use_cpu_cache

# Flickr, GCN, GAT
python3 main_ae.py \
    --model APPNP \
    --num_layer 2 \
    --repeat 3 \
    --num_epoch 200 \
    --lr 0.01 \
    --wd 5e-3 \
    --gpu 3 \
    --data_dir ../data \
    --dataset Flickr \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --graph_learn \
    --lr_gl 0.001 \
    --wd_gl 5e-3 \
    --thres_min_deg 10 \
    --thres_min_deg_ratio 1.0 \
    --window [10000,10000] \
    --shuffle [False,False] \
    --drop_last [False,False] \
    --epoch_train_gl 10 \
    --epoch_finetune_gl 3 \
    --k 6 \
    --cat_self False \
    --prunning False \
    --thres_prunning 0. \
    --use_cpu_cache




# Cora
python3 main_ae.py \
    --model GCN \
    --num_layer 2 \
    --repeat 3 \
    --num_epoch 200 \
    --lr 0.01 \
    --wd 5e-3 \
    --gpu 0 \
    --data_dir ../data \
    --dataset Cora \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --seed 0 \
    --graph_learn \
    --lr_gl 0.001 \
    --wd_gl 5e-3 \
    --thres_min_deg 3 \
    --thres_min_deg_ratio 1.0 \
    --window [10000,10000] \
    --shuffle [False,False] \
    --drop_last [False,False] \
    --epoch_train_gl 200 \
    --epoch_finetune_gl 30 \
    --seed_gl 0 \
    --k 8 \
    --cat_self True \
    --prunning False \
    --thres_prunning 0.

# CiteSeer
python3 main_ae.py \
    --model GraphSAGE \
    --num_layer 2 \
    --repeat 3 \
    --num_epoch 200 \
    --lr 0.01 \
    --wd 5e-3 \
    --gpu 1 \
    --data_dir ../data \
    --dataset CiteSeer \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --seed 0 \
    --graph_learn \
    --lr_gl 0.001 \
    --wd_gl 5e-3 \
    --thres_min_deg 3 \
    --thres_min_deg_ratio 0.8 \
    --window [10000,10000] \
    --shuffle [False,False] \
    --drop_last [False,False] \
    --epoch_train_gl 200 \
    --epoch_finetune_gl 30 \
    --seed_gl 0 \
    --k 8 \
    --cat_self False \
    --prunning False \
    --thres_prunning 0. \

# PubMed
# layer_num = 64, wd = 1e-4 for GCN2
python3 main_ae.py \
    --model GCN \
    --num_layer 2 \
    --repeat 3 \
    --num_epoch 200 \
    --lr 0.01 \
    --wd 5e-3 \
    --gpu 0 \
    --data_dir ../data \
    --dataset PubMed \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --seed 0 \
    --graph_learn \
    --lr_gl 0.001 \
    --wd_gl 5e-3 \
    --thres_min_deg 2 \
    --thres_min_deg_ratio 1.0 \
    --window [10000,10000] \
    --shuffle [True,True] \
    --drop_last [False,False] \
    --epoch_train_gl 30 \
    --epoch_finetune_gl 30 \
    --seed_gl 0 \
    --k 3 \
    --epsilon 0.9 \
    --cat_self True \
    --prunning False \
    --thres_prunning 0. \
    --use_cpu_cache \

# Cornell baseline wd=5e-3 for h2gcn, gprgnn
python3 main_ae.py \
    --model GraphSAGE \
    --num_layer 2 \
    --repeat 3 \
    --num_epoch 200 \
    --lr 0.01 \
    --wd 5e-3 \
    --gpu 2 \
    --data_dir ../data \
    --dataset Cornell \
    --moment 1 \
    --hidden 64 \
    --use_center_moment \
    --seed 0 \