# SwitchViT 联邦学习实验

当前分支以 **SwitchViT** 作为默认图像分类模型。模型由 `base.build_switch_vit_model` 构建，模型结构参数统一放在 `model_cfg` 中；当前只保留 SwitchViT 模型配置。

## 环境与启动

```bash
conda activate fl_moe
```

普通客户端等权 FedAvg 基线：

```bash

CUDA_VISIBLE_DEVICES=1 python algorithms/uniform.py \
    --dataset cifar10 \
    --num-clients 5 \
    --alpha 0.1 \
    --dirichlet-sample-balance unbalanced \
    --rounds 50 \
    --local-epochs 5 \
    --batch-size 64 \
    --data-augmentation true \
    --num-experts 4 \
    --transformer-layers 6 \
    --hidden-dim 256 \
    --attention-heads 4 \
    --patch-size 4 \
    --expert-hidden-dim 1024 \
    --moe-block-indices 2 4 6 \
    --dropout 0.0 \
    --aux-loss-coef 0.0 \
    --capacity-factor 0.0 \
    --token-dropping false \
    --optimizer sgd \
    --lr 0.01 \
    --momentum 0.9 \
    --weight-decay 0.0005 \
    --seed 0 \
    --output-dir outputs/qingcai
```


## 当前算法

- `algorithms/uniform.py`：客户端等权聚合，作为 FedAvg 基线。
- `algorithms/rfedags.py`：全客户端参与的 Euclidean 专家梯度流聚合版本。
- `algorithms/base.py`：负责公共配置、SwitchViT 构建、客户端训练、服务端协调、日志和结果输出。

当前 RFedAGS 版本要求：

- `frac = 1.0`，每轮全部客户端参与；
- 所有客户端每轮执行相同的本地更新步数；
- 使用 SGD，且 `momentum = 0`、`weight_decay = 0`；
- 不启用梯度裁剪。

在这些条件下，当前纯欧氏梯度流版本与客户端等权 FedAvg 的更新结果等价。保留该实现是为了明确区分客户端梯度流接口，并为后续独立设计流形优化方法保留算法扩展位置；当前版本不包含流形约束、投影或回缩。

## 本地训练预算

- Epoch 模式：设置 `local_epochs`，并令 `local_steps` 为 `null`。
- 固定步数模式：使用 `--local-steps K`；该设置优先于 `local_epochs`。
- RFedAGS 要求所有客户端的实际本地更新步数一致。使用不平衡样本划分时，建议显式设置统一的 `--local-steps K`。

## 结果绘图

```bash
python tools/paper_draw.py \
  --input-dir outputs/ \
  --dataset cifar10 \
  --architecture SwitchViT \
  --window 5 \
  --output-dir pictures/
```

绘图工具从 `config_used.yaml` 中读取 `model.architecture`，不再读取旧模型类型字段。

## 启动脚本

- `tools/clients.sh`：并行运行客户端数量对比实验。
- `tools/steps.sh`：并行运行本地更新步数对比实验。
- `tools/run.sh`：运行当前分支的基础算法对比实验。
