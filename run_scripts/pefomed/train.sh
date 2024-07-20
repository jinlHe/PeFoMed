#!/usr/bin/env bash

#train stage 1
torchrun -m torch.distributed.run --nproc_per_node=4 --master-port 1268 train.py \
--cfg-path pefomed/projects/pefomed/finetune_stage1.yaml \
--options model.ckpt="path_to_checkpoint.pth" \
run.output_dir="path_to_output" > output1.txt 2>&1

#train vqarad stage 2
torchrun -m torch.distributed.run --nproc_per_node=4 --master-port 1268 train.py \
--cfg-path pefomed/projects/pefomed/vqarad_finetune_stage2.yaml \
--options model.ckpt="path_to_checkpoint.pth" \
run.output_dir="path_to_output" > output2.txt 2>&1

 #train slake stage 2
 torchrun -m torch.distributed.run --nproc_per_node=4 --master-port 1268 train.py \
--cfg-path pefomed/projects/pefomed/slake_finetune_stage2.yaml \
--options model.ckpt="path_to_checkpoint.pth" \
run.output_dir="path_to_output" > output3.txt 2>&1

 #train pathvqa stage 2
 torchrun -m torch.distributed.run --nproc_per_node=4 --master-port 1268 train.py \
--cfg-path pefomed/projects/pefomed/pathvqa_finetune_stage2.yaml \
--options model.ckpt="path_to_checkpoint.pth" \
run.output_dir="path_to_output" > output4.txt 2>&1

 #train iuxray stage 2
 torchrun -m torch.distributed.run --nproc_per_node=4 --master-port 1268 train.py \
--cfg-path pefomed/projects/pefomed/iuxray_finetune_stage2.yaml \
--options model.ckpt="path_to_checkpoint.pth" \
run.output_dir="path_to_output" > output5.txt 2>&1
