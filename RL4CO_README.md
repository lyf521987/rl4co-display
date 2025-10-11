# RL4CO Flask 集成使用说明

## 📝 简介

本项目已成功集成真实的 RL4CO（Reinforcement Learning for Combinatorial Optimization）强化学习代码，支持通过 Web 界面进行交互式训练和可视化。

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**主要依赖包括：**
- Flask：Web 框架
- flask-mysqldb：数据库连接
- torch：深度学习框架
- rl4co：强化学习组合优化库
- lightning：PyTorch Lightning 训练框架
- matplotlib：可视化库

### 2. 配置数据库

确保 `config.py` 中的数据库配置正确：

```python
MYSQL_HOST = 'localhost'
MYSQL_USER = 'your_username'
MYSQL_PASSWORD = 'your_password'
MYSQL_DB = 'your_database'
```

### 3. 启动应用

```bash
python app.py
```

应用将在 `http://127.0.0.1:5000` 启动。

## 🎯 功能特性

### 真实训练模式

当 RL4CO 库正确安装时，系统将使用真实的强化学习训练：

1. **支持的模型**：
   - Attention Model (AM)
   - POMO
   - SymNCO
   - MatNet
   - MDAM
   - DeepACO

2. **支持的问题类型**：
   - TSP (Traveling Salesman Problem)
   - CVRP (Capacitated Vehicle Routing Problem)
   - SDVRP、PCTSP、OP、MDPP 等

3. **可配置参数**：
   - 训练轮数 (Epochs)
   - 批次大小 (Batch Size)
   - 学习率 (Learning Rate)
   - 问题规模（默认 50 个节点）

### 实时进度监控

- **进度条**：实时显示训练进度百分比
- **指标更新**：每个 epoch 更新 Loss 和 Reward
- **训练日志**：滚动显示详细的训练信息
- **最佳指标追踪**：记录训练过程中的最佳 Reward

### 结果可视化

训练完成后自动生成：
1. **路径对比图**：显示训练前后的路径质量对比
2. **指标统计**：最终 Loss、Reward 和最佳 Reward
3. **检查点保存**：自动保存模型权重，支持继续训练

## 📂 目录结构

```
Flask-Login-Register-Demo/
├── app.py                    # 主应用文件（包含真实 RL4CO 代码）
├── config.py                 # 配置文件
├── requirements.txt          # 依赖列表
├── templates/
│   ├── login.html           # 登录页面
│   ├── res.html             # 注册页面
│   └── index.html           # 主界面（训练配置和结果展示）
├── static/
│   ├── css/
│   └── model_plots/         # 训练结果可视化图片（自动生成）
└── checkpoints/             # 模型检查点（自动生成）
```

## 🔧 技术架构

### 后端集成

```python
# 1. 自定义 Lightning Callback 捕获训练进度
class ProgressCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # 实时推送训练指标到前端
        ...

# 2. 真实 RL4CO 训练函数
def real_rl4co_training(config, session_id):
    # 初始化环境
    env = TSPEnv(generator_params={'num_loc': 50})
    
    # 定义策略网络
    policy = AttentionModelPolicy(...)
    
    # 定义 RL 模型
    model = REINFORCE(env, policy, ...)
    
    # 训练
    trainer = RL4COTrainer(...)
    trainer.fit(model)
    
    # 生成可视化结果
    ...
```

### 实时通信

- **Server-Sent Events (SSE)**：单向实时数据流
- **消息队列**：线程安全的进度传递
- **后台训练**：不阻塞 Flask 主线程

## 📊 使用流程

1. **登录/注册**：访问系统
2. **配置参数**：
   - 选择模型（如 Attention Model）
   - 选择问题类型（如 TSP）
   - 选择训练策略（如 REINFORCE）
   - 设置训练轮数、批次大小、学习率
3. **开始训练**：点击"开始训练"按钮
4. **实时监控**：观察进度条、指标和日志
5. **查看结果**：训练完成后查看可视化对比图和统计数据

## ⚙️ 配置说明

### GPU 支持

系统自动检测并使用 GPU（如果可用）：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
```

### 检查点管理

- 自动保存：训练完成后保存到 `checkpoints/` 目录
- 自动加载：如果存在同名检查点，将从检查点继续训练
- 命名规则：`{problem_type}-{model_type}.ckpt`

### 问题规模

默认设置为 50 个节点，可以在 `app.py` 中修改：

```python
num_loc = 50  # 问题规模
env = TSPEnv(generator_params={'num_loc': num_loc})
```

## 🐛 故障排除

### RL4CO 未安装

如果 RL4CO 库未安装，系统会自动切换到**模拟训练模式**：
- 不会进行真实训练
- 使用随机生成的指标模拟训练过程
- 在控制台会显示警告信息

解决方法：
```bash
pip install rl4co
```

### CUDA 内存不足

如果遇到 GPU 内存不足，可以：
1. 减小批次大小（Batch Size）
2. 减小问题规模（num_loc）
3. 使用 CPU 训练（自动检测）

### 图片不显示

确保：
1. `static/model_plots/` 目录存在（自动创建）
2. Flask static 文件路由正常工作
3. 浏览器能访问 `/static/model_plots/` 路径

## 📈 性能优化建议

1. **使用 GPU**：训练速度提升 10-50 倍
2. **适当的 Batch Size**：
   - GPU：512-1024
   - CPU：128-256
3. **合理的训练轮数**：
   - 快速测试：3-5 epochs
   - 正常训练：50-100 epochs
   - 完整训练：500+ epochs

## 🔐 安全注意事项

1. 训练过程在后台线程中运行，不会阻塞主应用
2. 每个训练会话有唯一的 session_id
3. 训练状态存储在内存中（重启应用会丢失）
4. 模型检查点和图片会持久化保存

## 📚 参考资源

- [RL4CO 官方文档](https://github.com/ai4co/rl4co)
- [PyTorch Lightning 文档](https://lightning.ai/docs/pytorch/)
- [Flask 文档](https://flask.palletsprojects.com/)

## 🎓 山西大学

本项目由山西大学计算机科学与技术学院维护，用于强化学习教学和科研。

---

**版本**：1.0.0  
**最后更新**：2025-01-11


