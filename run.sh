#!/bin/bash

workspace_dir=''

export CUDA_DEVICE_ORDER=PCI_BUS_ID

method_name=$1
config_name=$2
shift
shift

num_workers=4
pin_memory=true
wandb_offline=false
do_sweep=false
enable_debug=false
disable_ib=false
declare -a mixin_config

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -R|--resume) resume_file_path="${2}"; shift ;;
        -W|--workers) num_workers=$2; shift ;;
        --output_dir) workspace_dir=$2; shift ;;
        --checkpoint_interval) checkpoint_interval=$2; shift ;;
        --no_pin_memory) pin_memory=false ;;
        --device_ids) device_ids="$2"; shift ;;
        --offline) wandb_offline=true ;;
        --do_sweep) do_sweep=true ;;
        --alt_sweep_config) sweep_config_path="${2}"; shift ;;
        --sweep_run_times) sweep_run_times=$2; shift ;;
        --sweep_id) sweep_id="$2"; shift ;;
        --run_id) run_id="$2"; shift ;;
        --mixin) mixin_config+=("$2"); shift ;;
        --nnodes) NUM_NODES=$2; shift ;;
        --node_rank) NODE_RANK=$2; shift ;;
        --master_address) MASTER_ADDRESS="$2"; shift ;;
        --seed) seed=$2; shift ;;
        --debug) enable_debug=true; shift ;;
        --disable_ib) disable_ib=true ;;
        --weight_path) weight_path="$2"; shift ;;
        --evaluation_only) mixin_config+=("/mixin/evaluation.yaml") ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$workspace_dir" ]]; then
    echo "workspace_dir cannot be empty"
    exit 1
fi

set -o pipefail

if [[ -z "$device_ids" ]]
then
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    nvidia-smi
else
    num_gpus=$(nvidia-smi -i "$device_ids" --query-gpu=name --format=csv,noheader | wc -l)
    nvidia-smi -i "$device_ids"
    export CUDA_VISIBLE_DEVICES="$device_ids"
fi

if [[ -z "$run_id" ]]; then
    if [[ -z "$DATE_WITH_TIME" ]]; then
        DATE_WITH_TIME=$(date "+%Y.%m.%d-%H.%M.%S-%6N")
    fi
    run_id="$config_name"
    for i in "${mixin_config[@]}"
    do
        i=$(basename -- "$i")
        run_id="$run_id-mixin-${i%.*}"
    done
    run_id="$run_id-$DATE_WITH_TIME"
fi

output_dir="$workspace_dir/$run_id"
mkdir -p "$output_dir"

export OMP_NUM_THREADS=1

target_options=("$method_name" "$config_name")

common_options=("--run_id" "$run_id" "--output_dir" "$workspace_dir" "--num_workers" "$num_workers")

if [[ "$pin_memory" == true ]]; then
    common_options+=("--pin_memory")
fi
if [[ "$enable_debug" == true ]]; then
    common_options+=("--debug")
fi
if [[ "$disable_ib" == true ]]; then
    export NCCL_IB_DISABLE=1
fi
if [[ -n "$checkpoint_interval" ]]; then
    common_options+=("--checkpoint_interval" "$checkpoint_interval")
fi
if [[ "$wandb_offline" == true ]]; then
    common_options+=("--wandb_run_offline")
fi
if [[ -n "$NUM_NODES" ]]; then
    common_options+=("--distributed_nnodes" "$NUM_NODES")
fi
if [[ -n "$MASTER_ADDRESS" ]]; then
    common_options+=("--master_address" "$MASTER_ADDRESS")
fi
if [[ -n "$seed" ]]; then
    common_options+=("--seed" "$seed")
fi
if [[ -n "$weight_path" ]]; then
    common_options+=("--weight_path" "$weight_path")
fi
if [[ "$num_gpus" -gt 1 ]]; then
    common_options+=("--distributed_nproc_per_node" "$num_gpus")
fi
if [[ -n "$NODE_RANK" ]]; then
    common_options+=("--distributed_node_rank" "$NODE_RANK")
fi
if [[ -n "$NUM_NODES" || "$num_gpus" -gt 1 ]]; then
    common_options+=("--distributed_do_spawn_workers")
fi
for i in "${mixin_config[@]}"
do
    common_options+=("--mixin_config" "$i")
done
if [[ -n "$resume_file_path" ]]; then
    common_options+=("--resume" "$resume_file_path")
fi

if [[ "$do_sweep" == false ]]; then
    output_log="$output_dir/train_stdout.log"
    if [[ -n "$NODE_RANK" ]]; then
        output_log="$output_dir/train_stdout.$NODE_RANK.log"
    fi
    python main.py "${target_options[@]}" "${common_options[@]}" |& tee -a "$output_log"
else
    if [[ -n "$NUM_NODES" && "$NUM_NODES" -gt 1 ]]; then
        echo "Multi-nodes distributed training currently not support for hyper-parameter tunning"
        exit 1
    fi
    output_log="$output_dir/sweep_stdout.log"
    if [[ -n "$NODE_RANK" ]]; then
        output_log="$output_dir/sweep_stdout.$NODE_RANK.log"
    fi
    sweep_options=()
    if [[ -n "$sweep_config_path" ]]; then
        sweep_options+=("--sweep_config" "$sweep_config_path")
    fi
    if [[ -n "$sweep_id" ]]; then
        sweep_options+=("--sweep_id" "$sweep_id")
    fi
    if [[ -n "$sweep_run_times" ]]; then
        sweep_options+=("--agents_run_limit" "$sweep_run_times")
    fi

    python sweep.py "${target_options[@]}" "${sweep_options[@]}" "${common_options[@]}" |& tee -a "$output_log"
fi
