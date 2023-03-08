gpus=("0" "1" "2" "3");
for gpu in ${gpus}; do
    COMM="CUDA_VISIBLE_DEVICES=${gpu} & python3 sweep-agent.py";
    bash -c "echo CUDA_VISIBLE_DEVICES=${gpu}; python3 sweep-agent.py";
done