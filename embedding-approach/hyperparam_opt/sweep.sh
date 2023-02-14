gpus=("0" "1");
apikey="abc
for gpu in ${gpus}; do
    echo ${gpu};
    COMM="CUDA_VISIBLE_DEVICES=${gpu} & python3 sweep-agent.py";
    bash -c "echo CUDA_VISIBLE_DEVICES=${gpu}; 
    {echo '2'; echo ${apikey}} | python3 sweep-agent.py";
done