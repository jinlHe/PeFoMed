#!/usr/bin/env bash

 #eval vqa
torchrun -m torch.distributed.run --nproc_per_node=2 --master-port 1268 evaluate.py \
--cfg-path pefomed/projects/pefomed/vqarad_finetune_stage2.yaml \
--options model.ckpt="checkpoint.pth" \
run.batch_size_eval=64 \
run.output_dir="" > output.txt 2>&1

 #eval report
torchrun -m torch.distributed.run --nproc_per_node=2 --master-port 1268 evaluate.py \
--cfg-path pefomed/projects/pefomed/iuxray_finetune_stage2.yaml \
--options model.ckpt="checkpoint.pth" \
run.batch_size_eval=64 \
run.output_dir="" > output.txt 2>&1