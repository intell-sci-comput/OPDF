#!/bin/bash
# sst2
nohup bash -c 'CUDA_VISIBLE_DEVICES=0 python run_glue_MPO_losslayer.py --model_name_or_path /mnt/name/data/nlp_data/theseus/BertFineTrain/download/sst2/bert-base-uncased-finetuned-sst2/ --task_name sst-2 --do_train --do_eval --do_lower_case --data_dir /mnt/name/data/nlp_data/GLUE/SST-2 --max_seq_length 128 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --learning_rate 1e-6 --save_steps 50 --num_train_epochs 20 --output_dir /mnt/name/data/checkpoint/nlp/theseus/save_successor/ --evaluate_during_training --replacing_rate 0.3 --scheduler_type linear --scheduler_linear_k 0.0003 --overwrite_output_dir' >log/sst2_mpo.log >&1 &
wait
nohup bash -c 'CUDA_VISIBLE_DEVICES=2 python run_glue_MPO_losslayer.py --model_name_or_path /mnt/name/data/nlp_data/theseus/BertFineTrain/download/sst2/bert-base-uncased-finetuned-sst2/ --task_name sst-2 --do_train --do_eval --do_lower_case --data_dir /mnt/name/data/nlp_data/GLUE/SST-2 --max_seq_length 128 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --learning_rate 3e-6 --save_steps 50 --num_train_epochs 20 --output_dir /mnt/name/data/checkpoint/nlp/theseus/save_successor/ --evaluate_during_training --replacing_rate 0.3 --scheduler_type linear --scheduler_linear_k 0.0003 --overwrite_output_dir' >log/sst2_mpo_2.log >&1 &
wait
nohup bash -c 'CUDA_VISIBLE_DEVICES=3 python run_glue_MPO_losslayer.py --model_name_or_path /mnt/name/data/nlp_data/theseus/BertFineTrain/download/sst2/bert-base-uncased-finetuned-sst2/ --task_name sst-2 --do_train --do_eval --do_lower_case --data_dir /mnt/name/data/nlp_data/GLUE/SST-2 --max_seq_length 128 --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 32 --learning_rate 6e-6 --save_steps 50 --num_train_epochs 20 --output_dir /mnt/name/data/checkpoint/nlp/theseus/save_successor/ --evaluate_during_training --replacing_rate 0.3 --scheduler_type linear --scheduler_linear_k 0.0003 --overwrite_output_dir' >log/sst2_mpo_3.log >&1 &
wait



