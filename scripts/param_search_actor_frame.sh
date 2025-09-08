#!/bin/bash

timestamp=$(date +"%Y%m%d_%H%M%S")

best_acc=0.0
best_params=""

log_file="param_search_results_actor_frame.txt"
echo "Frame Method Parameter Search Results for Actor Dataset (Lorentz Model)" > $log_file
echo "----------------------------------------" >> $log_file
echo "Started at: $(date)" >> $log_file
echo "----------------------------------------" >> $log_file

dims=(8 12 16)  
num_layers=(2 3)  
dropouts=(0.05 0.1 0.2)  
kernel_sizes=(3 4)  
kp_extents=(0.6 0.66)  
lrs=(0.002 0.005)  
weight_decays=(0.0001 0.0002)  
lr_reduce_freqs=(200 250)  
gammas=(0.5 0.6)  
batch_sizes=(16 24)  
patience_values=(100 150)  
linear_before_values=(16 24)  
temperatures=(0.2 0.3) 
frame_sample_ratios=(0.2 0.3 0.4)  
seeds=(42 1234)  
grad_accumulation_steps=(2 4)
progress_file="actor_frame_progress_${timestamp}.txt"
echo "0" > $progress_file

for i in {1..2}; do
    
    echo $i > $progress_file
    
    dim=${dims[$RANDOM % ${#dims[@]}]}
    num_layer=${num_layers[$RANDOM % ${#num_layers[@]}]}
    dropout=${dropouts[$RANDOM % ${#dropouts[@]}]}
    kernel_size=${kernel_sizes[$RANDOM % ${#kernel_sizes[@]}]}
    kp_extent=${kp_extents[$RANDOM % ${#kp_extents[@]}]}
    lr=${lrs[$RANDOM % ${#lrs[@]}]}
    weight_decay=${weight_decays[$RANDOM % ${#weight_decays[@]}]}
    lr_reduce_freq=${lr_reduce_freqs[$RANDOM % ${#lr_reduce_freqs[@]}]}
    gamma=${gammas[$RANDOM % ${#gammas[@]}]}
    batch_size=${batch_sizes[$RANDOM % ${#batch_sizes[@]}]}
    patience=${patience_values[$RANDOM % ${#patience_values[@]}]}
    linear_before=${linear_before_values[$RANDOM % ${#linear_before_values[@]}]}
    temperature=${temperatures[$RANDOM % ${#temperatures[@]}]}
    frame_sample_ratio=${frame_sample_ratios[$RANDOM % ${#frame_sample_ratios[@]}]}
    seed=${seeds[$RANDOM % ${#seeds[@]}]}
    grad_accum=${grad_accumulation_steps[$RANDOM % ${#grad_accumulation_steps[@]}]}
    
    echo "Running configuration $i/2..."
    echo "----------------------------------------"
    echo "dim: $dim"
    echo "num_layers: $num_layer"
    echo "dropout: $dropout"
    echo "kernel_size: $kernel_size"
    echo "KP_extent: $kp_extent"
    echo "lr: $lr"
    echo "weight_decay: $weight_decay"
    echo "lr_reduce_freq: $lr_reduce_freq"
    echo "gamma: $gamma"
    echo "batch_size: $batch_size"
    echo "patience: $patience"
    echo "linear_before: $linear_before"
    echo "temperature: $temperature"
    echo "frame_sample_ratio: $frame_sample_ratio"
    echo "seed: $seed"
    echo "grad_accumulation_steps: $grad_accum"
    echo "----------------------------------------"

    temp_output="temp_output_actor_frame_$i.txt"
    
    python train.py \
        --task nc \
        --dataset film \
        --model HKPNet \
        --manifold Lorentz \
        --dim $dim \
        --num_layers $num_layer \
        --epochs 5000 \
        --lr $lr \
        --dropout $dropout \
        --cuda 6 \
        --weight_decay $weight_decay \
        --optimizer radam \
        --momentum 0.999 \
        --patience $patience \
        --seed $seed \
        --log_freq 10 \
        --eval_freq 10 \
        --save 1 \
        --save_dir "logs/actor_frame" \
        --lr_reduce_freq $lr_reduce_freq \
        --gamma $gamma \
        --min_epochs 100 \
        --use_geoopt True \
        --AggKlein False \
        --corr 0 \
        --nei_agg 0 \
        --bias 1 \
        --act relu \
        --n_heads 4 \
        --alpha 0.2 \
        --double_precision 1 \
        --use_att 0 \
        --local_agg 0 \
        --kernel_size $kernel_size \
        --KP_extent $kp_extent \
        --radius 1 \
        --deformable False \
        --linear_before $linear_before \
        --batch_size $batch_size \
        --val_prop 0.05 \
        --test_prop 0.1 \
        --use_feats 1 \
        --normalize_feats 1 \
        --normalize_adj 1 \
        --split_seed $seed \
        --split_graph False \
        --use_frame True \
        --temperature $temperature \
        --frame_sample_ratio $frame_sample_ratio \
        --use_amp \
        --grad_accumulation_steps $grad_accum \
        --c 1.0 --r 2.0 --t 1.0 --margin 2.0 \
        2>&1 | tee $temp_output

    test_acc=$(grep "Test set results:" $temp_output | grep -oP "test_acc: \K[0-9.]+")
    
    echo "Configuration $i:" >> $log_file
    echo "Parameters:" >> $log_file
    echo "dim: $dim" >> $log_file
    echo "num_layers: $num_layer" >> $log_file
    echo "dropout: $dropout" >> $log_file
    echo "kernel_size: $kernel_size" >> $log_file
    echo "KP_extent: $kp_extent" >> $log_file
    echo "lr: $lr" >> $log_file
    echo "weight_decay: $weight_decay" >> $log_file
    echo "lr_reduce_freq: $lr_reduce_freq" >> $log_file
    echo "gamma: $gamma" >> $log_file
    echo "batch_size: $batch_size" >> $log_file
    echo "patience: $patience" >> $log_file
    echo "linear_before: $linear_before" >> $log_file
    echo "temperature: $temperature" >> $log_file
    echo "frame_sample_ratio: $frame_sample_ratio" >> $log_file
    echo "seed: $seed" >> $log_file
    echo "grad_accumulation_steps: $grad_accum" >> $log_file
    echo "Test Accuracy: $test_acc" >> $log_file
    echo "----------------------------------------" >> $log_file

    if (( $(echo "$test_acc > $best_acc" | bc -l) )); then
        best_acc=$test_acc
        best_params="dim: $dim, num_layers: $num_layer, dropout: $dropout, kernel_size: $kernel_size, KP_extent: $kp_extent, lr: $lr, weight_decay: $weight_decay, lr_reduce_freq: $lr_reduce_freq, gamma: $gamma, batch_size: $batch_size, patience: $patience, linear_before: $linear_before, temperature: $temperature, frame_sample_ratio: $frame_sample_ratio, seed: $seed, grad_accumulation_steps: $grad_accum"
        
        echo "New best accuracy: $best_acc"
        echo "Best parameters: $best_params"
        
        echo "Best Configuration (acc: $best_acc):" > best_params_actor_frame.txt
        echo "$best_params" >> best_params_actor_frame.txt
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