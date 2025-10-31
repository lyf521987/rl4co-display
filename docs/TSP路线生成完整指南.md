# TSP路线生成 - 完整指南

> 详细说明TSP路线生成的完整过程，从问题输入到最终路线输出，包括原理、流程和技术细节。

---

## 📋 目录

1. [快速参考](#快速参考)
2. [核心原理](#核心原理)
3. [完整流程](#完整流程)
4. [技术细节](#技术细节)
5. [代码示例](#代码示例)

---

## 快速参考

### TSP问题定义

**Traveling Salesman Problem (旅行商问题)**

给定N个城市的坐标，找到一条最短的路径，访问每个城市恰好一次并返回起点。

### 输入输出

**输入**:
```python
城市坐标: [(x1, y1), (x2, y2), ..., (xN, yN)]
例如: [(0.1, 0.2), (0.3, 0.5), (0.8, 0.6), ...]
```

**输出**:
```python
访问顺序: [0, 3, 7, 12, 5, ..., 1]  # 城市索引
总成本: 10.990  # 总路径长度
```

### 快速流程

```
1. 输入城市坐标
   ↓
2. 初始化策略网络
   ↓
3. 生成路线（Greedy解码）
   ↓
4. 计算总成本
   ↓
5. 可视化结果
```

### 关键参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| **城市数量** | N个城市 | 50 |
| **坐标范围** | x, y ∈ [0, 1] | 归一化坐标 |
| **解码方式** | Greedy/Sampling | Greedy |
| **训练轮数** | Epochs | 10 |

---

## 核心原理

### 1. 问题表示

#### 城市坐标矩阵
```
城市坐标 (N × 2):
[
  [0.1234, 0.5678],  # 城市0
  [0.9012, 0.3456],  # 城市1
  [0.7890, 0.2345],  # 城市2
  ...
]
```

#### 距离计算
```
欧几里得距离:
d(i, j) = √[(xi - xj)² + (yi - yj)²]
```

#### 路径成本
```
总成本 = Σ d(路径[i], 路径[i+1]) + d(路径[-1], 路径[0])
       = 所有边的距离之和 + 回到起点的距离
```

### 2. 策略网络

#### 注意力机制（Attention Model）

```
输入: 城市坐标 → Embedding → Context
      ↓
   Encoder (Transformer)
      ↓
   Decoder (逐步选择城市)
      ↓
   输出: 访问顺序
```

**核心思想**:
- 用注意力机制学习城市之间的关系
- 逐步选择下一个要访问的城市
- 考虑已访问和未访问的状态

#### POMO（多起点优化）

```
基本思想: 同时从多个起点开始构建路线
优势: 增加解的多样性，提升质量

单次前向传播 → N条不同的路线
选择最优: min(cost1, cost2, ..., costN)
```

### 3. 解码方式

#### Greedy解码（贪心）
```python
def greedy_decode():
    路线 = [起点]
    while 还有未访问城市:
        计算到所有未访问城市的概率
        选择概率最高的城市
        添加到路线
    return 路线

特点:
- 速度最快（<1秒）
- 确定性输出
- 质量中等
```

#### Sampling解码（采样）
```python
def sampling_decode(M=1280):
    路线集合 = []
    for i in range(M):
        根据概率分布采样选择城市
        生成一条路线
        路线集合.append(路线)
    return 最优路线

特点:
- 速度中等（20秒-2分钟）
- 随机性输出
- 质量较好
```

### 4. 训练过程

#### REINFORCE算法
```
1. 生成路线（策略网络）
2. 计算奖励（负的总成本）
3. 计算梯度
4. 更新网络参数

目标: 最大化期望奖励
    = 最小化期望成本
```

#### 训练与推理对比

| 阶段 | 目的 | 方式 | 时间 |
|------|------|------|------|
| **训练** | 学习策略 | Sampling | 5-20分钟 |
| **推理** | 生成路线 | Greedy | <1秒 |

---

## 完整流程

### 阶段1: 数据准备

```python
# 1. 生成随机TSP实例
import numpy as np

num_cities = 50
city_coords = np.random.rand(num_cities, 2)

# 结果：
# [[0.1234, 0.5678],
#  [0.9012, 0.3456],
#  ...]

print(f"生成了 {num_cities} 个随机城市")
```

### 阶段2: 模型初始化

```python
# 2. 加载训练好的模型
from rl4co.models import AttentionModel

model = AttentionModel.load_from_checkpoint("checkpoints/tsp-attention.ckpt")
model.eval()  # 设置为评估模式

print("模型加载完成")
```

### 阶段3: 路线生成

```python
# 3. 使用Greedy解码生成路线
td_init = env.reset(batch_size=[3])  # 3个测试实例

with torch.no_grad():
    out = model(td_init, decode_type="greedy")

actions = out['actions']  # 访问顺序
rewards = out['reward']   # 总成本（负值）

print(f"路线: {actions[0].cpu().numpy()}")
print(f"成本: {-rewards[0].item():.4f}")
```

### 阶段4: 成本计算

```python
# 4. 计算路线的总成本
def calculate_route_cost(coords, route):
    total_cost = 0
    for i in range(len(route)):
        start = coords[route[i]]
        end = coords[route[(i + 1) % len(route)]]
        distance = np.sqrt(np.sum((start - end) ** 2))
        total_cost += distance
    return total_cost

cost = calculate_route_cost(city_coords, actions[0])
print(f"总成本: {cost:.4f}")
```

### 阶段5: 结果可视化

```python
# 5. 绘制路线图
import matplotlib.pyplot as plt

def plot_route(coords, route, title):
    plt.figure(figsize=(8, 8))
    
    # 绘制城市点
    plt.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=3)
    
    # 绘制路径
    for i in range(len(route)):
        start = coords[route[i]]
        end = coords[route[(i + 1) % len(route)]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=2)
    
    # 高亮起点
    plt.scatter(coords[route[0], 0], coords[route[0], 1], 
               c='green', s=200, marker='*', zorder=4)
    
    plt.title(title, fontsize=14)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.show()

plot_route(city_coords, actions[0], f"TSP路线 | 成本={cost:.3f}")
```

---

## 技术细节

### 1. 策略网络架构

```
Attention Model 架构:

输入层 (N × 2)
   ↓
Embedding层 (N × 128)
   ↓
Multi-Head Attention × L layers
   ↓ 
Context Vector (128)
   ↓
Decoder (自回归)
   ↓
输出: 城市选择概率 (N)
```

**参数量**: ~10M（百万级）

### 2. 训练细节

```python
# 训练配置
{
    "model": "attention",
    "problem": "tsp",
    "optimizer": "Adam",
    "learning_rate": 1e-4,
    "batch_size": 512,
    "epochs": 10,
    "num_cities": 50,
    "decode_type": "sampling"  # 训练时用采样
}

# 训练流程
for epoch in range(epochs):
    for batch in dataloader:
        # 生成路线（采样）
        out = model(batch, decode_type="sampling")
        
        # 计算loss（REINFORCE）
        loss = compute_reinforce_loss(out)
        
        # 反向传播
        loss.backward()
        optimizer.step()
```

### 3. 解码策略对比

| 解码方式 | 速度 | 质量 | 确定性 | 用途 |
|---------|------|------|--------|------|
| **Greedy** | 最快 | 中等 | 确定 | 推理部署 |
| **Sampling** | 中等 | 较好 | 随机 | 训练过程 |
| **Beam Search** | 慢 | 最好 | 确定 | 离线优化 |

### 4. 性能优化技巧

#### 数据增强
```python
# 8种对称变换
transformations = [
    original,
    flip_x,
    flip_y,
    flip_xy,
    rotate_90,
    rotate_180,
    rotate_270,
    transpose
]

# 对每个变换生成路线，选最优
best_route = min(routes, key=lambda r: r.cost)
```

#### 多起点（POMO）
```python
# 从每个城市作为起点
routes = []
for start_city in range(num_cities):
    route = model.decode(start_city)
    routes.append(route)

best_route = min(routes, key=lambda r: r.cost)
```

---

## 代码示例

### 完整示例：端到端TSP求解

```python
"""
完整的TSP求解示例
从问题生成到结果可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from rl4co.models import AttentionModel
from rl4co.envs import TSPEnv

# ============ 1. 准备数据 ============
print("步骤1: 准备TSP数据")
num_cities = 50
batch_size = 1

# 创建环境
env = TSPEnv()
td_init = env.reset(batch_size=[batch_size])

# 获取城市坐标
city_coords = td_init['locs'][0].cpu().numpy()  # (50, 2)
print(f"生成了 {num_cities} 个城市")

# ============ 2. 加载模型 ============
print("\n步骤2: 加载训练好的模型")
model = AttentionModel.load_from_checkpoint(
    "checkpoints/tsp-attention.ckpt"
)
model.eval()
print("模型加载完成")

# ============ 3. 生成路线 ============
print("\n步骤3: 生成TSP路线")

# 方式1: Greedy解码（快速）
with torch.no_grad():
    out_greedy = model(td_init, decode_type="greedy")
    
route_greedy = out_greedy['actions'][0].cpu().numpy()
cost_greedy = -out_greedy['reward'][0].item()

print(f"Greedy路线: {route_greedy}")
print(f"Greedy成本: {cost_greedy:.4f}")

# ============ 4. 可视化结果 ============
print("\n步骤4: 可视化路线")

def plot_tsp_route(coords, route, title, cost):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 绘制城市点
    ax.scatter(coords[:, 0], coords[:, 1], 
              c='lightblue', s=200, zorder=3, 
              edgecolors='black', linewidths=2)
    
    # 标注城市编号
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, str(i), 
               fontsize=8, ha='center', va='center',
               fontweight='bold')
    
    # 绘制路径
    for i in range(len(route)):
        start = coords[route[i]]
        end = coords[route[(i + 1) % len(route)]]
        ax.plot([start[0], end[0]], [start[1], end[1]], 
               'b-', linewidth=2, alpha=0.6, zorder=1)
        
        # 添加箭头
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        ax.arrow(mid_x - dx*0.1, mid_y - dy*0.1, 
                dx*0.2, dy*0.2,
                head_width=0.02, head_length=0.01,
                fc='blue', ec='blue', alpha=0.6)
    
    # 高亮起点
    ax.scatter(coords[route[0], 0], coords[route[0], 1],
              c='green', s=400, marker='*', zorder=4,
              edgecolors='darkgreen', linewidths=2,
              label='起点')
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_title(f"{title}\n总成本: {cost:.4f}", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('tsp_route.png', dpi=150, bbox_inches='tight')
    print("图片已保存: tsp_route.png")

plot_tsp_route(city_coords, route_greedy, 
              "TSP最优路线", cost_greedy)

print("\n完成！")
```

### 输出示例

```
步骤1: 准备TSP数据
生成了 50 个城市

步骤2: 加载训练好的模型
模型加载完成

步骤3: 生成TSP路线
Greedy路线: [ 0 24 32 18 41  3 22 26 15 48 35  1 46 34 38 20 43 13  9 11  6 27  5
 10 30 29 39 44 19 31 12 33 17 16  2 25  4 23 40 42 36 49 14 45 21 47
 37  8  7 28]
Greedy成本: 5.7456

步骤4: 可视化路线
图片已保存: tsp_route.png

完成！
```

---

## 常见问题

### Q1: 为什么训练用Sampling，推理用Greedy？

**A**: 
- **训练**: 需要探索不同路线，Sampling提供随机性
- **推理**: 需要稳定输出，Greedy提供确定性

### Q2: 如何提升路线质量？

**A**: 
1. 增加训练轮数
2. 使用数据增强
3. 使用Sampling代替Greedy（慢但质量好）
4. 使用POMO多起点策略

### Q3: 成本如何计算？

**A**: 
```python
总成本 = 所有相邻城市之间的距离之和 + 回到起点的距离
```

### Q4: 可以处理多少个城市？

**A**: 
- 训练模型: 通常50个城市
- 泛化能力: 可处理100+城市
- 实际限制: 取决于GPU显存

---

## 相关文档

- [动态GIF功能完整指南](动态GIF功能完整指南.md)
- [实时训练曲线功能完整指南](实时训练曲线功能完整指南.md)
- [模型知识库功能完整指南](模型知识库功能完整指南.md)

---

**文档版本**: v1.0.0  
**最后更新**: 2024年  
**平台**: RL4CO 强化学习优化平台  
**单位**: 山西大学 计算机科学与技术学院

