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
vit_tiny

kill $(cat outputs/fashionmnist_vit_tiny/launcher_logs/*.pid)

模型与专家聚合算法的职责约定：
- \`model\` 只选择网络结构；当前统一使用 \`sparse_moe_classifier\`。
- 是否对专家参数施加正交约束，由专家聚合算法决定，不再提供 \`sparse_moe_orthogonal\` 模型。
- \`fedavg_orthogonal.py\` 与 \`rfedags_orthogonal.py\` 默认约束所有专家的 \`fc1.weight\` 行空间。
- 正交约束对象可在算法配置的 \`orthogonal.stiefel_blocks\` 中通过参数名模式调整。

正交 FedAvg：
\`\`\`bash
python algorithms/fedavg_orthogonal.py --dataset cifar10 --backbone resnet_cifar
\`\`\`

RFedAGS 正交专家聚合：
\`\`\`bash
python algorithms/rfedags_orthogonal.py --dataset cifar10 --backbone resnet_cifar --local-steps 5
\`\`\`
