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

# export CUDA_VISIBLE_DEVICES=6
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 2 --datasets 3  > 23.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=7
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 2 --datasets 4  > 24.log 2>&1 &


# export CUDA_VISIBLE_DEVICES=0
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 1 --datasets 5  > 15.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 1 --datasets 6  > 16.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 2 --datasets 5  > 25.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=3
# export PYTHONPATH=$PWD:$PYTHONPATH 
# nohup python train.py --task 2 --datasets 6  > 26.log 2>&1 &



export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PWD:$PYTHONPATH 
nohup python train.py --task 3 --datasets 1  > log/31.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PWD:$PYTHONPATH 
nohup python train.py --task 3 --datasets 2  > log/32.log 2>&1 &

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$PWD:$PYTHONPATH 
nohup python train.py --task 3 --datasets 3  > log/33.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$PWD:$PYTHONPATH 
nohup python train.py --task 3 --datasets 4  > log/34.log 2>&1 &

export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=$PWD:$PYTHONPATH 
nohup python train.py --task 3 --datasets 5  > log/35.log 2>&1 &

export CUDA_VISIBLE_DEVICES=5
export PYTHONPATH=$PWD:$PYTHONPATH 
nohup python train.py --task 3 --datasets 6  > log/36.log 2>&1 &