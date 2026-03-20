#!/bin/bash
# set_visible_device.sh

NGPUS=$(echo ${SLURM_STEP_GPUS//[^,]/} | wc -c)
# CUDA_VISIBLE_DEVICES = round_down(local_process_id_in_node / processes_per_gpu)
export CUDA_VISIBLE_DEVICES=$(( $SLURM_LOCALID * $SLURM_CPUS_PER_TASK * $NGPUS / ${SLURM_JOB_CPUS_PER_NODE%(*)} ))
exec $*
