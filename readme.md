tmux new -s train_clean1
tmux attach -t train_clean1

cd /home/cjq/Project/clean
conda activate fl_moe

单个实验运行命令
CUDA_VISIBLE_DEVICES=0 python algorithms/uniform.py \
  --dataset cifar10 \
  --backbone resnet_cifar \
  --model sparse_moe_classifier \
  --moe-hidden-dim 128 \
  --output-dir two_exp/new_moe_yes_fedavg

折线图运行命令
python tools/paper_draw.py \
  --input-dir outputs_tkfac/cifar10_resnet_cifar \
  --window 5 \
  --max-round 50 \
  --output-dir ./paper_pic/line

迭代步数折线运行命令
python tools/plot_tkfac_server_steps.py \
  --input-dir outputs_tkfac_server_steps/cifar10_resnet_cifar \
  --window 5 \
  --max-round 50 \
  --output-dir ./paper_pic/Steps_line

支持的数据集：
cifar10
cifar100
cinic10
fashionmnist
stl10
tiny-imagenet-200
femnist

支持的backbone：
resnet_cifar
vgg11
vit_tiny
swin_tiny

kill $(cat outputs/fashionmnist_vit_tiny/launcher_logs/*.pid)