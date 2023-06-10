export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export PATH=/cpfs01/user/guosizheng/anaconda3/bin:$PATH

cd /cpfs01/user/guosizheng/union_train_sam3D/
source ../.bashrc
conda activate medsam
python train.py --task_name union_train_11click_test --click_type random --multi_click --gpu_ids 0 1 --multi_gpu