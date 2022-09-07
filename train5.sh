# export CUDA_VISIBLE_DEVICES=0
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 1 --datasets 5  > 5.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 1 --datasets 6  > 6.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$PWD:$PYTHONPATH 
nohup python train.py --task 2 --datasets 5  > 25.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PWD:$PYTHONPATH 
nohup python train.py --task 2 --datasets 6  > 26.log 2>&1 &