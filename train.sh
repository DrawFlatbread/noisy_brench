# export CUDA_VISIBLE_DEVICES=0
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 1 --datasets 1  > 11.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 1 --datasets 2  > 12.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 1 --datasets 3  > 13.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=3
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 1 --datasets 4  > 14.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=4
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 2 --datasets 1  > 21.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=5
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 2 --datasets 2  > 22.log 2>&1 &

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=$PWD:$PYTHONPATH 
nohup python train.py --task 2 --datasets 3  > 23.log 2>&1 &

export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=$PWD:$PYTHONPATH 
nohup python train.py --task 2 --datasets 4  > 24.log 2>&1 &


# export CUDA_VISIBLE_DEVICES=0
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 1 --datasets 5  > 5.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 1 --datasets 6  > 6.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 2 --datasets 5  > 25.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=3
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 2 --datasets 6  > 26.log 2>&1 &