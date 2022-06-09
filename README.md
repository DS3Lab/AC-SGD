# AC-SGD

## Setup:

- Create environment:

  ```bash
  conda create -n acsgd python=3.8
  
  conda activate acsgd
  ```

- Install PyTorch env: 

  ```bash
  pip3 install torch==1.9.0+cu111 torchtext -f https://download.pytorch.org/whl/torch_stable.html

  # Magic, not sure why cupy-cuda111 would not work, it seems that cupy-cuda111 will use different PTX from torch.
  pip3 install cupy-cuda110==8.6.0
  ```
  
- Other dependencies:
 
  ```bash
  pip3 install datasets==2.2.2
  pip3 install transformers==4.19.2
  pip3 install sentencepiece==0.1.96 # required by deberta
  ```
  
- Setup network configuration:

  ```bash
  export GLOO_SOCKET_IFNAME=ens3

  export NCCL_SOCKET_IFNAME=ens3
  ```
  
- Download datasets:

  ```bash
  wget https://gpt-activation-compression.s3.us-east-2.amazonaws.com/data.zip
  
  unzip data.zip
  ```

## Run Distributed Gpipe:

- Partition the pre-trained model:
  
  ```bash
  # gpt2
  python convert_gpt2_checkpoint --model-name gpt2-xl --save-dir checkpoints/
      
  # deberta 
  python convert_deberta_checkpoint --model-name deberta-v2-xxl --save-dir checkpoints/
  ```

- On each node, run:
  
  ```bash
  python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 0 --rank i # (i=0,...,N-1)
      
  python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 0 --rank i # (i=0,...,N-1)
  ```
  where "ARGS" contains training-related configurations, which should remain the same across nodes. An example could be:
  ```bash
  ARGS="--model-name checkpoints/gpt2-xl \
    --tokenizer-name gpt2-xl \
    --load-pretrained-model true \
    --task-name wikitext --n-epochs 10 --warmup-epochs 1 \
    --num-layers 6 --num-heads 25 --embedding-dim 1600 \
    --num-iters 10000000 --lr 5e-5 --seq-length 1024 --batch-size 32 --micro-batch-size 1 \
    --forward-compress-method delta \
    --forward-bits 4 \
    --backward-compress-method fixpoint \
    --backward-bits 8 \
    --dist-url tcp://XXX.XXX.XXX.XXX:9000 \
    --world-size N --pipeline-group-size N \
    --pp-mode gpipe --profiling no-profiling --do-evaluation true"
  ```
  Remember to modify "--dist-url", "--world-size" and "--pipeline-group-size" before running.
  
  Complete examples can be found "./run_lm.sh" and "./run_deberta.sh", which simulate on a single machine with 8 GPUs.