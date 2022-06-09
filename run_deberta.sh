
# Important! Pre-trained weights need to be partioned before fine-tuning.
python convert_deberta_checkpoint.py --model-name microsoft/deberta-v2-xxlarge --save-dir checkpoints

ARGS="--model-name ./checkpoints/deberta-v2-xxl \
--tokenizer-name microsoft/deberta-v2-xxlarge \
--load-pretrained-model true --seed 42 \
--task-name cola --n-epochs 10 --warmup-steps 50 --warmup-epochs 1 \
--num-layers 6 --num-heads 24 --embedding-dim 1536 \
--num-iters 10000000 --lr 2.5e-6 --seq-length 256 --batch-size 32 --micro-batch-size 8 \
--forward-compress-method delta \
--forward-bits 3 \
--backward-compress-method fixpoint \
--backward-bits 6 \
--dist-url tcp://127.0.0.1:9042 \
--world-size 8 --pipeline-group-size 8 \
--pp-mode gpipe --profiling no-profiling --do-evaluation true"

(trap 'kill 0' SIGINT; \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    & \
python dist_deberta_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)
