#!/bin/bash

timestamp=$(date +"%Y%m%d_%H%M%S")

best_acc=0.0
best_params=""

log_file="param_search_results_cornell.txt"
echo "Parameter Search Results for Cornell Dataset" > $log_file
echo "----------------------------------------" >> $log_file
echo "Started at: $(date)" >> $log_file
echo "----------------------------------------" >> $log_file

dims=(10 12 14 16 18)  
num_layers=(2)  
dropouts=(0.1 0.15 0.2 0.25)  
kernel_sizes=(6 8 10)  
kp_extents=(0.55 0.6 0.65 0.7)  
lrs=(0.008 0.01 0.012 0.015)  
weight_decays=(0.00003 0.00005 0.0001 0.0002)  
lr_reduce_freqs=(140 150 160 170)  
gammas=(0.75 0.8 0.85 0.9)  
temperatures=(0.2 0.25 0.3 0.35 0.4)  

progress_file="cornell_progress_${timestamp}.txt"
echo "0" > $progress_file

for i in {1..2}; do
    echo $i > $progress_file
    
    dim=${dims[$RANDOM % ${#dims[@]}]}
    dropout=${dropouts[$RANDOM % ${#dropouts[@]}]}
    kernel_size=${kernel_sizes[$RANDOM % ${#kernel_sizes[@]}]}
    kp_extent=${kp_extents[$RANDOM % ${#kp_extents[@]}]}
    lr=${lrs[$RANDOM % ${#lrs[@]}]}
    weight_decay=${weight_decays[$RANDOM % ${#weight_decays[@]}]}
    lr_reduce_freq=${lr_reduce_freqs[$RANDOM % ${#lr_reduce_freqs[@]}]}
    gamma=${gammas[$RANDOM % ${#gammas[@]}]}
    temperature=${temperatures[$RANDOM % ${#temperatures[@]}]}
    
    echo "Running configuration $i/2..."
    echo "----------------------------------------"
    echo "dim: $dim"
    echo "num_layers: 2"
    echo "dropout: $dropout"
    echo "kernel_size: $kernel_size"
    echo "KP_extent: $kp_extent"
    echo "lr: $lr"
    echo "weight_decay: $weight_decay"
    echo "lr_reduce_freq: $lr_reduce_freq"
    echo "gamma: $gamma"
    echo "temperature: $temperature"
    echo "----------------------------------------"

    temp_output="temp_output_cornell_opt_$i.txt"
    
    python train.py \
        --task nc \
        --dataset cornell \
        --model HKPNet \
        --manifold Lorentz \
        --dim $dim \
        --num_layers 2 \
        --epochs 5000 \
        --use_frame True \
        --lr $lr \
        --dropout $dropout \
        --cuda 2 \
        --weight_decay $weight_decay \
        --optimizer radam \
        --momentum 0.999 \
        --patience 200 \
        --seed 42 \
        --log_freq 10 \
        --eval_freq 10 \
        --save 0 \
        --lr_reduce_freq $lr_reduce_freq \
        --gamma $gamma \
        --min_epochs 100 \
        --use_geoopt True \
        --bias 1 \
        --act relu \
        --use_feats 1 \
        --normalize_feats 1 \
        --normalize_adj 1 \
        --split_seed 23 \
        --corr 0 \
        --temperature $temperature \
        --kernel_size $kernel_size \
        --KP_extent $kp_extent 2>&1 | tee $temp_output

    rotation_info=$(cat $temp_output | grep -E "Rotation matrix|rotation_matrix")
    echo "Rotation matrix info:" >> $log_file
    echo "$rotation_info" >> $log_file

    test_acc=$(cat $temp_output | grep "Test set results:" | grep -oP "test_acc: \K[0-9.]+")
    
    echo "Configuration $i:" >> $log_file
    echo "Parameters:" >> $log_file
    echo "dim: $dim" >> $log_file
    echo "num_layers: 2" >> $log_file
    echo "dropout: $dropout" >> $log_file
    echo "kernel_size: $kernel_size" >> $log_file
    echo "KP_extent: $kp_extent" >> $log_file
    echo "lr: $lr" >> $log_file
    echo "weight_decay: $weight_decay" >> $log_file
    echo "lr_reduce_freq: $lr_reduce_freq" >> $log_file
    echo "gamma: $gamma" >> $log_file
    echo "temperature: $temperature" >> $log_file
    echo "Test Accuracy: $test_acc" >> $log_file
    echo "----------------------------------------" >> $log_file

    if (( $(echo "$test_acc > $best_acc" | bc -l) )); then
        best_acc=$test_acc
        best_params="dim: $dim, num_layers: 2, dropout: $dropout, kernel_size: $kernel_size, KP_extent: $kp_extent, lr: $lr, weight_decay: $weight_decay, lr_reduce_freq: $lr_reduce_freq, gamma: $gamma, temperature: $temperature"
        
        echo "New best accuracy: $best_acc"
        echo "Best parameters: $best_params"
        
        echo "Best Configuration (acc: $best_acc):" > best_params_cornell.txt
        echo "$best_params" >> best_params_cornell.txt
    fi

    rm $temp_output
done

echo "----------------------------------------" >> $log_file
echo "Search completed at: $(date)" >> $log_file
echo "Best accuracy: $best_acc" >> $log_file
echo "Best parameters: $best_params" >> $log_file

rm $progress_file

echo "Parameter search completed!"
echo "Best accuracy: $best_acc"
echo "Best parameters: $best_params"
echo "Full results saved in $log_file" 