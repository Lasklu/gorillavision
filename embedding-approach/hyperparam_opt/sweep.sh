gpus=("0" "1" "2" "3");
source keyfile
for gpu in ${gpus}; do
    COMM="CUDA_VISIBLE_DEVICES=${gpu} & python3 sweep-agent.py";
    PROMT="2\n${APIKEY}"
    bash -c "echo CUDA_VISIBLE_DEVICES=${gpu}; printf ${PROMT} | python3 sweep-agent.py";
done