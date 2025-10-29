# TSP路线生成原理详解

## 📋 概述

在RL4CO平台中，TSP（旅行商问题）的路线生成是通过训练好的深度强化学习策略网络来完成的。本文档详细说明整个生成和可视化过程。

---

## 🔄 完整流程图

```
训练完成
   ↓
1. 准备策略网络和测试环境
   ↓
2. 生成测试用的TSP实例（城市坐标）
   ↓
3. 使用随机策略生成基准路线（对比用）
   ↓
4. 使用训练好的策略生成优化路线
   ↓
5. 计算两种路线的总成本
   ↓
6. 可视化对比并保存图片
   ↓
7. 返回图片路径给前端展示
```

---

## 📝 详细步骤解析

### 步骤 1：准备策略网络 (第355-356行)

```python
policy = model.policy.to(device)  # 将训练好的策略网络移到GPU/CPU
td_init = env.reset(batch_size=[3]).to(device)  # 生成3个TSP测试实例
```

**说明：**
- `policy`：这是训练好的注意力模型（Attention Model）策略网络
- `td_init`：包含3个随机生成的TSP实例，每个实例有50个城市（默认配置）
- 每个城市用二维坐标 (x, y) 表示，坐标值在 [0, 1] 范围内

**数据结构示例：**
```python
td_init = {
    'locs': tensor([[[0.2, 0.8], [0.5, 0.3], ...], # 问题1：50个城市坐标
                    [[0.1, 0.6], [0.9, 0.2], ...], # 问题2：50个城市坐标
                    [[0.7, 0.4], [0.3, 0.9], ...]]) # 问题3：50个城市坐标
}
```

---

### 步骤 2：生成随机策略路线（基准对比）(第358-361行)

```python
out_untrained = policy(
    td_init.clone(),           # 输入：城市坐标
    phase="test",              # 测试模式（不更新参数）
    decode_type="sampling",    # 解码方式：采样（随机选择）
    return_actions=True        # 返回访问顺序
)
actions_untrained = out_untrained['actions'].cpu().detach()  # 访问序列
rewards_untrained = out_untrained['reward'].cpu().detach()   # 路线成本（负值）
```

**解码策略 - Sampling（采样）：**
- 在每一步，根据策略网络输出的概率分布**随机采样**下一个要访问的城市
- 模拟未经训练或随机决策的效果
- 路线质量较差，作为对比基准

**输出示例：**
```python
actions_untrained[0] = [0, 15, 23, 8, 42, 31, ...]  # 访问城市的顺序
rewards_untrained[0] = -12.5834  # 总路径长度的负值（RL中reward越大越好）
```

---

### 步骤 3：生成训练后的优化路线 (第363-366行)

```python
out_trained = policy(
    td_init.clone(),           # 输入：相同的城市坐标
    phase="test",              # 测试模式
    decode_type="greedy",      # 解码方式：贪心（选择最优）
    return_actions=True        # 返回访问顺序
)
actions_trained = out_trained['actions'].cpu().detach()   # 优化后的访问序列
rewards_trained = out_trained['reward'].cpu().detach()    # 优化后的路线成本
```

**解码策略 - Greedy（贪心）：**
- 在每一步，**选择概率最大**的城市作为下一个访问点
- 利用训练好的策略网络的知识
- 生成质量较高的路线

**输出示例：**
```python
actions_trained[0] = [0, 3, 7, 12, 19, 25, ...]  # 优化后的访问顺序
rewards_trained[0] = -8.2341  # 更短的总路径长度
```

**关键区别：**
```
随机策略（Sampling）：Cost = 12.58
训练后策略（Greedy）：Cost = 8.23
改进幅度：(12.58 - 8.23) / 12.58 ≈ 34.6% 提升！
```

---

### 步骤 4：路线生成的核心机制

#### 4.1 注意力机制（Attention Model）

策略网络使用注意力机制来决定下一个访问哪个城市：

```
当前状态 = [已访问的城市, 当前位置, 未访问的城市坐标]
          ↓
      编码器（Encoder）
          ↓
      注意力层（Attention）
          ↓
      解码器（Decoder）
          ↓
    输出概率分布：[城市1: 0.05, 城市2: 0.32, 城市3: 0.15, ...]
          ↓
    Greedy: 选择城市2（概率最大0.32）
    Sampling: 按概率随机选择
```

#### 4.2 逐步构建路线

TSP路线是逐个城市选择构建的：

```python
初始状态：当前位置 = 城市0（起点）
        已访问 = [0]
        未访问 = [1, 2, 3, ..., 49]

第1步：策略网络输出 → 选择城市3
      已访问 = [0, 3]
      未访问 = [1, 2, 4, 5, ..., 49]

第2步：策略网络输出 → 选择城市7
      已访问 = [0, 3, 7]
      未访问 = [1, 2, 4, 5, 6, 8, ..., 49]

...重复49次...

第50步：访问最后一个城市，返回起点
       完整路线 = [0, 3, 7, 12, ..., 0]
```

---

### 步骤 5：可视化生成 (第368-381行)

```python
for i, td in enumerate(td_init):  # 遍历3个测试实例
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 创建左右对比图
    
    # 左图：随机策略
    env.render(td, actions_untrained[i], ax=axs[0])
    axs[0].set_title(f"Random | Cost = {-rewards_untrained[i].item():.3f}")
    
    # 右图：训练后策略
    env.render(td, actions_trained[i], ax=axs[1])
    axs[1].set_title(f"Trained | Cost = {-rewards_trained[i].item():.3f}")
    
    # 保存图片
    plot_filename = f"comparison_{session_id[:8]}_{i+1}.png"
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
```

**env.render() 函数的工作原理：**

1. **绘制城市点**：在二维平面上画出所有城市的坐标
2. **绘制路径**：按照 actions 的顺序用线连接城市
3. **标注起点**：通常用特殊颜色标记起点城市
4. **显示方向**：用箭头表示访问顺序

**可视化示例：**
```
左图（Random）：
  City 0 → City 15 → City 23 → ... → City 0
  路径交叉较多，总长度：12.58

右图（Trained）：
  City 0 → City 3 → City 7 → ... → City 0
  路径较为平滑，总长度：8.23
```

---

## 🎯 关键技术点

### 1. **解码类型对比**

| 解码方式 | 原理 | 用途 | 路线质量 |
|---------|------|------|---------|
| **Sampling** | 按概率随机采样 | 探索、基准对比 | 较差 |
| **Greedy** | 选择概率最大的动作 | 快速推理 | 较好 |
| **Beam Search** | 保留多个候选路径 | 更优质量（未使用） | 最好 |

### 2. **Reward 计算**

```python
# TSP的reward是路径总长度的负值
total_distance = 0
for i in range(len(actions) - 1):
    city_a = locs[actions[i]]
    city_b = locs[actions[i+1]]
    distance = sqrt((city_a[0] - city_b[0])^2 + (city_a[1] - city_b[1])^2)
    total_distance += distance

reward = -total_distance  # 负值，因为要最小化距离
```

### 3. **批量处理**

```python
td_init = env.reset(batch_size=[3])  # 同时生成3个问题
```

- 一次处理3个不同的TSP实例
- 利用GPU并行计算加速
- 生成3张对比图展示不同案例

---

## 📊 输出结果

### 生成的图片

每次训练完成后生成**3张对比图**：

```
static/model_plots/
  ├── comparison_6c61a541_1.png  # 问题1的对比
  ├── comparison_6c61a541_2.png  # 问题2的对比
  └── comparison_6c61a541_3.png  # 问题3的对比
```

### 图片内容

```
┌─────────────────────────────────────────────┐
│  Random | Cost = 12.58  │  Trained | Cost = 8.23  │
│                         │                          │
│    ●─────●              │      ●───●               │
│    │  ╱  │              │      │   │               │
│    ● ●  ●               │      ●───●               │
│    交叉路径              │      优化路径             │
└─────────────────────────────────────────────┘
```

---

## 🔬 为什么训练后的路线更好？

### 1. **学习到的模式**
- 策略网络通过10000个训练样本学习
- 学会识别"近邻优先"、"避免交叉"等启发式规则
- 注意力机制学会关注哪些城市应该优先访问

### 2. **REINFORCE算法的作用**
```
好的路线（短距离）→ 高reward → 增加该策略的概率
差的路线（长距离）→ 低reward → 降低该策略的概率
```

### 3. **编码器-解码器架构**
- **编码器**：理解所有城市的全局分布
- **注意力层**：捕捉城市之间的相对关系
- **解码器**：基于当前状态做出最优决策

---

## 💡 实际应用示例

假设有以下TSP实例：

```python
城市坐标：
City 0: (0.1, 0.2) - 起点
City 1: (0.15, 0.25) - 距离起点很近
City 2: (0.8, 0.9) - 距离起点很远
City 3: (0.12, 0.18) - 距离起点很近
...
```

**随机策略可能选择：**
```
0 → 2 → 1 → 3 → ...  (先去远处，再回来，效率低)
```

**训练后策略会选择：**
```
0 → 1 → 3 → ... → 2  (先访问附近城市，最后去远处)
```

---

## 🎓 总结

TSP路线生成过程：

1. **输入**：50个城市的二维坐标
2. **处理**：策略网络逐步决策访问顺序
3. **解码**：Greedy选择最优动作，Sampling模拟随机
4. **计算**：统计路径总长度作为成本
5. **可视化**：用matplotlib绘制路径对比图
6. **输出**：保存为PNG图片供前端展示

**核心优势：**
- ✅ 端到端学习，无需手工设计规则
- ✅ 推理速度快（几毫秒生成一条路线）
- ✅ 泛化能力强（可应用于不同规模的问题）
- ✅ 可扩展到其他组合优化问题（VRP、CVRP等）

---

**技术栈：**
- PyTorch：深度学习框架
- Lightning：训练流程管理
- RL4CO：强化学习组合优化库
- Matplotlib：可视化绘图

