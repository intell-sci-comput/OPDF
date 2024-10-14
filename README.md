# OPDF

## Introduction

we propose a general **O**ver-**P**arameterization **D**istillation **F**ramework, namely **OPDF**, to improve the performance of knowledge distillation. Given the parameter matrices of a student model, we first over-parameterize them through MPO decomposition and then utilize high-order tensor alignment losses to ensure efficient information transfer. Orginal paper is "**Over-parameterized Student Model via Tensor Decomposition Boosted Knowledge Distillation"**.

![avatar](resources/main.png)

### Theseus

#### Install the dependencies

```bash
conda create -n theseus python=3.8

conda activate theseus

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

cd BERT-of-Theseus

pip install -r requirements.txt

```

#### Usage

#### Training

You can adjust the over-parameterization scale by modifying the variables *input3072\_size*, *input768\_size, input3072\_size2* and *input768\_size2* in *BERT-of-Theseus/bert\_of\_theseus/modeling\_bert\_of\_theseus.py*. Detailed methods can be found in the paper.

```bash
cd BERT-of-Theseus
# SST-2
nohup bash glue_script/script.sh &
```

### LGTM

#### Install the dependencies

```bash
conda create -n lgtm python=3.8

conda activate lgtm

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

cd BERT-of-Theseus

pip install -r requirements.txt
```

#### Usage

#### Training

You can adjust over-parameterization scale by modifying the variables *input3072\_size* and *input768\_size* in *LGTM/run\_glue\_mpo\_laterloss.py*. Detailed methods can be found in the paper.

```bash
cd LGTM
# MRPC
nohup bash -c 'CUDA_VISIBLE_DEVICES=1 taskset -c 9-17 python run_glue_mpo_laterloss.py --model_name_or_path student_model_path --teacher_model teacher_model_path --task_name mrpc --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --learning_rate 1e-06 --t_learning_rate 3e-05 --alpha_kd 1.0 --temperature 1.0 --num_train_epochs 15 --output_dir mrpc_output_path --eval_steps 5 --do_train --do_eval --train_teacher --init_classifier_to_zero --use_lgtm --overwrite_output_dir >log/mrpc_mpo.log' >&1 &
```

