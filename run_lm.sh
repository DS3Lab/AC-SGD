
# Important! Pre-trained weights need to be partioned before fine-tuning.
python convert_gpt2_checkpoint.py --model-name gpt2-xl --save-dir checkpoints

ARGS="--model-name checkpoints/gpt2-xl \
--tokenizer-name gpt2-xl \
--load-pretrained-model true \
--task-name arxiv21 --n-epochs 10 --warmup-epochs 1 \
--num-layers 6 --num-heads 25 --embedding-dim 1600 \
--num-iters 10000000 --lr 5e-5 --seq-length 1024 --batch-size 32 --micro-batch-size 1 \
--forward-compress-method delta \
--forward-bits 4 \
--backward-compress-method fixpoint \
--backward-bits 8 \
--dist-url tcp://127.0.0.1:9033 \
--world-size 8 --pipeline-group-size 8 \
--pp-mode gpipe --profiling no-profiling --do-evaluation true"

(trap 'kill 0' SIGINT; \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
    & \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
    & \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    & \
python dist_lm_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
wait)

# > /dev/null 2>&1 &
