# RL4CO Display - 强化学习优化可视化平台

> 山西大学 计算机科学与技术学院  
> 基于 RL4CO 的交互式强化学习训练与可视化平台

---

## 📝 项目简介

RL4CO Display 是一个集成了真实 RL4CO（Reinforcement Learning for Combinatorial Optimization）强化学习框架的 Web 可视化平台，支持通过浏览器进行交互式模型训练、实时监控和结果可视化。

### 核心特性

✅ **多用户系统** - 完整的用户认证和数据隔离机制  
✅ **真实训练** - 集成 RL4CO 库，支持多种强化学习算法  
✅ **实时监控** - 实时显示训练进度、Loss 和 Reward 曲线  
✅ **结果可视化** - 自动生成路径对比图和动态 GIF  
✅ **多模型支持** - 20+ 种强化学习模型可选  
✅ **多问题类型** - 支持 TSP、CVRP、OP、PCTSP、PDP 等问题  
✅ **文件管理** - 完整的训练文件管理和下载功能

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- MySQL 8.0+
- CUDA (可选，用于 GPU 加速)

### 步骤 1: 克隆项目

```bash
git clone https://github.com/your-repo/rl4co-display.git
cd rl4co-display
```

### 步骤 2: 安装依赖

```bash
pip install -r requirements.txt
```

**主要依赖包括：**
- Flask 3.0.0 - Web 框架
- Flask-MySQLdb 2.0.0 - 数据库连接
- torch >= 2.0.0 - 深度学习框架
- rl4co >= 0.4.0 - 强化学习组合优化库
- lightning >= 2.0.0 - PyTorch Lightning 训练框架
- matplotlib >= 3.7.0 - 可视化库

### 步骤 3: 配置数据库

1. **创建数据库**

```bash
mysql -u root -p
```

```sql
CREATE DATABASE flaskdemo_user CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

2. **初始化数据库表**

```bash
mysql -u root -p flaskdemo_user < database_init_with_auth.sql
```

3. **配置数据库连接**

编辑 `config.py` 文件：

```python
MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PASSWORD = 'your_password'
MYSQL_DB = 'flaskdemo_user'
```

> 📖 **详细配置说明**: 完整的配置指南请参考 [CONFIG.md](CONFIG.md)，包括安全配置、环境变量使用等最佳实践。

### 步骤 4: 启动应用

```bash
python app.py
```

应用将在 `http://127.0.0.1:5000` 启动。

### 步骤 5: 注册并登录

1. 访问 `http://localhost:5000/register` 注册新用户
2. 登录后即可开始训练模型

---

## 🎯 功能特性

### 1. 用户认证与数据隔离

- **安全注册登录** - 密码使用 pbkdf2:sha256 加密存储
- **Session 管理** - 基于 Flask-Login 的用户会话管理
- **数据严格隔离** - 每个用户只能访问自己的训练数据和文件
- **权限控制** - 未登录用户无法访问训练功能
- **安全防护** - SQL 注入、路径遍历等攻击防护

### 2. 强化学习训练

#### 支持的模型

| 类别 | 模型 |
|------|------|
| **基础模型** | Attention Model (AM), POMO |
| **改进模型** | SymNCO, MatNet, MDAM |
| **ACO 系列** | DeepACO |
| **其他** | 20+ 种模型 |

#### 支持的问题类型

- **TSP** (Traveling Salesman Problem) - 旅行商问题
- **CVRP** (Capacitated Vehicle Routing Problem) - 带容量约束的车辆路径问题
- **OP** (Orienteering Problem) - 定向问题
- **PCTSP** (Prize Collecting TSP) - 带奖励的旅行商问题
- **PDP** (Pickup and Delivery Problem) - 取送货问题
- **SDVRP、MDPP** 等更多问题类型

#### 可配置参数

- 训练轮数 (Epochs)
- 批次大小 (Batch Size)
- 学习率 (Learning Rate)
- 问题规模（节点数量）
- 训练策略（REINFORCE、A2C、PPO 等）

### 3. 实时监控

- **进度条** - 实时显示训练进度百分比
- **实时曲线** - Loss 和 Reward 随 epoch 变化的动态曲线
- **训练日志** - 滚动显示详细的训练信息
- **最佳指标追踪** - 记录训练过程中的最佳性能

### 4. 结果可视化

#### 静态对比图
- 训练前后路径质量对比
- 成本变化统计
- 多样本展示

#### 动态 GIF
- 逐帧显示路径构建过程
- 实时显示成本变化
- 支持多个样本对比

#### 训练曲线
- Loss 和 Reward 变化曲线
- 多指标对比
- 可导出图片

### 5. 文件管理

- **文件列表** - 查看所有训练生成的文件
- **分类显示** - 按类型（图片、GIF、检查点）分类
- **下载功能** - 一键下载训练结果
- **批量删除** - 支持批量清理文件

### 6. 算法对比

- **Benchmark 数据** - 基于 KDD 2025 论文的算法性能对比
- **交互式图表** - 使用 ECharts 展示性能数据
- **多问题规模** - 支持 20 节点和 50 节点对比

### 7. 模型知识库

- **详细信息** - 每个模型的论文出处、发表年份
- **技术架构** - 模型的核心技术和创新点
- **性能数据** - 各模型在标准测试集上的性能
- **使用建议** - 模型选择和参数配置建议

---

## 📂 项目结构

```
rl4co-display/
├── app.py                          # 主应用文件
├── auth_module.py                  # 用户认证模块
├── config.py                       # 配置文件
├── requirements.txt                # Python 依赖
├── database_init_with_auth.sql    # 数据库初始化脚本
│
├── templates/                      # HTML 模板
│   ├── index.html                 # 训练主页面
│   ├── login.html                 # 登录页面
│   ├── register.html              # 注册页面
│   ├── res.html                   # 训练结果页面
│   ├── file_manager.html          # 文件管理页面
│   ├── benchmark.html             # 算法对比页面
│   ├── model_info.html            # 模型知识库页面
│   └── profile.html               # 用户账户页面
│
├── static/                         # 静态资源
│   ├── css/                       # 样式文件
│   │   ├── login.css
│   │   └── navigation.css
│   ├── js/                        # JavaScript 文件
│   │   └── navigation.js
│   ├── img/                       # 图片资源
│   └── model_plots/               # 训练结果（自动生成）
│       └── user_X/                # 按用户隔离
│           ├── comparison_*.png
│           ├── animation_*.gif
│           └── training_curves_*.png
│
├── checkpoints/                    # 模型检查点（自动生成）
│   └── user_X/                    # 按用户隔离
│       └── *.ckpt
│
├── lightning_logs/                 # PyTorch Lightning 日志
│   └── version_X/
│
└── docs/                           # 完整文档
    ├── README.md                  # 文档索引
    ├── TSP路线生成完整指南.md
    ├── 实时训练曲线功能完整指南.md
    ├── 动态GIF功能完整指南.md
    ├── 文件管理功能完整指南.md
    ├── 算法对比页面功能完整指南.md
    ├── 模型知识库功能完整指南.md
    └── 数据库清理完整指南.md
```

---

## 🔧 技术架构

### 后端技术栈

- **Flask 3.0** - Web 框架
- **Flask-Login** - 用户认证
- **MySQL** - 数据存储
- **PyTorch** - 深度学习框架
- **RL4CO** - 强化学习库
- **PyTorch Lightning** - 训练框架

### 前端技术栈

- **HTML5/CSS3** - 页面结构和样式
- **JavaScript (ES6+)** - 交互逻辑
- **ECharts** - 数据可视化
- **Server-Sent Events (SSE)** - 实时数据推送

### 核心实现

#### 1. 用户认证流程

```python
# 注册
用户注册 → 密码加密(pbkdf2:sha256) → 存入数据库 → 跳转登录

# 登录
用户登录 → 密码验证 → 创建 Session → 跳转主页

# 权限检查
@login_required 装饰器 → 检查 Session → 获取 user_id → 过滤数据
```

#### 2. 训练流程

```python
# 1. 配置训练参数
前端表单 → Flask 接收 → 验证参数

# 2. 启动训练
生成 session_id → 后台线程训练 → 不阻塞主线程

# 3. 实时推送
ProgressCallback → 捕获训练指标 → SSE 推送 → 前端更新

# 4. 生成结果
训练完成 → 生成可视化 → 保存文件 → 更新数据库
```

#### 3. 数据隔离机制

```python
# 文件路径隔离
/static/model_plots/user_{user_id}/filename.png

# 数据库隔离
SELECT * FROM training_files WHERE user_id = current_user.id

# API 权限检查
@login_required
def api():
    user_id = get_current_user_id()
    # 只返回该用户的数据
```

---

## 📊 数据库设计

### users 表 - 用户信息

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INT | 主键 |
| username | VARCHAR(50) | 用户名（唯一） |
| password | VARCHAR(255) | 密码（加密） |
| email | VARCHAR(100) | 邮箱 |
| created_at | TIMESTAMP | 创建时间 |

### training_sessions 表 - 训练会话

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INT | 主键 |
| session_id | VARCHAR(64) | 会话ID（唯一） |
| user_id | INT | 用户ID（外键） |
| model_type | VARCHAR(50) | 模型类型 |
| problem_type | VARCHAR(50) | 问题类型 |
| status | VARCHAR(20) | 训练状态 |
| config | JSON | 训练配置 |
| results | JSON | 训练结果 |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

### training_files 表 - 训练文件

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INT | 主键 |
| file_id | VARCHAR(64) | 文件ID |
| session_id | VARCHAR(64) | 会话ID（外键） |
| user_id | INT | 用户ID（外键） |
| file_type | VARCHAR(20) | 文件类型 |
| file_path | VARCHAR(255) | 文件路径 |
| file_size | BIGINT | 文件大小 |
| created_at | TIMESTAMP | 创建时间 |

---

## 🎓 使用指南

### 新手入门

1. **注册账户** → 访问 `/register` 页面注册
2. **了解模型** → 访问"模型知识库"了解各模型特点
3. **配置训练** → 在主页配置训练参数
4. **开始训练** → 点击"开始训练"并实时监控
5. **查看结果** → 训练完成后查看可视化结果
6. **管理文件** → 在"文件管理"页面下载或删除文件

### 进阶使用

1. **算法对比** → 访问"算法对比"页面查看不同算法性能
2. **参数调优** → 尝试不同的学习率、批次大小等参数
3. **多问题训练** → 尝试不同的问题类型（TSP、CVRP 等）
4. **检查点复用** → 从之前的检查点继续训练

---

## 📚 详细文档

### 配置文档

- **[CONFIG.md](CONFIG.md)** - 配置文件完整指南（数据库、依赖、安全配置等）

### 功能文档

完整的功能文档位于 `docs/` 目录：

- [文档索引](docs/README.md) - 所有文档的导航
- [TSP路线生成完整指南](docs/TSP路线生成完整指南.md) - TSP 问题求解详解
- [实时训练曲线功能完整指南](docs/实时训练曲线功能完整指南.md) - 训练监控实现
- [动态GIF功能完整指南](docs/动态GIF功能完整指南.md) - 动态可视化实现
- [文件管理功能完整指南](docs/文件管理功能完整指南.md) - 文件管理系统
- [算法对比页面功能完整指南](docs/算法对比页面功能完整指南.md) - Benchmark 对比
- [模型知识库功能完整指南](docs/模型知识库功能完整指南.md) - 模型详细信息
- [数据库清理完整指南](docs/数据库清理完整指南.md) - 数据库管理

### 部署文档

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - 生产环境部署指南

---

## 🔐 安全机制

### 1. 密码安全
- 使用 `pbkdf2:sha256` 加密算法
- 加盐哈希存储
- 不可逆加密

### 2. Session 安全
- 使用 Flask 的 SECRET_KEY
- HttpOnly Cookie
- Session 过期机制

### 3. SQL 注入防护
- 使用参数化查询
- 输入验证和清理
- ORM 安全实践

### 4. 路径遍历防护
- 文件路径白名单
- `safe_join_path` 函数
- 用户目录隔离

### 5. XSS 防护
- 自动 HTML 转义
- CSP 策略
- 输入验证

---

## ⚙️ 配置说明

> 📖 **完整配置指南**: 详细的配置文件说明请参考 [CONFIG.md](CONFIG.md)

### 快速配置

配置文件位置和说明：

- **config.py** - 数据库连接配置，包含MySQL主机、用户名、密码等
- **requirements.txt** - Python依赖包列表
- **database_init_with_auth.sql** - 数据库初始化脚本

详细配置说明请查看 [CONFIG.md](CONFIG.md)。

### GPU 加速

系统自动检测并使用 GPU（如果可用）：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

如果遇到 GPU 内存不足：
1. 减小批次大小 (Batch Size)
2. 减小问题规模 (num_loc)
3. 使用 CPU 训练

### 检查点管理

- **自动保存** - 训练完成后保存到 `checkpoints/user_X/` 目录
- **自动加载** - 如果存在同名检查点，可从检查点继续训练
- **命名规则** - `{problem_type}-{model_type}.ckpt`

### 性能优化建议

**批次大小 (Batch Size)**
- GPU: 512-1024
- CPU: 128-256

**训练轮数 (Epochs)**
- 快速测试: 3-5 epochs
- 正常训练: 50-100 epochs
- 完整训练: 500+ epochs

**学习率 (Learning Rate)**
- Attention Model: 1e-4
- POMO: 1e-4
- 其他模型: 参考模型知识库

---

## 🐛 故障排除

### 问题 1: 数据库连接失败

**症状**
```
✗ 警告: 认证模块初始化失败
```

**解决方法**
1. 检查 MySQL 是否运行: `mysql -u root -p`
2. 检查 `config.py` 中的数据库配置
3. 确认数据库存在: `SHOW DATABASES;`

### 问题 2: RL4CO 未安装

**症状**
```
WARNING: RL4CO not available, using mock training
```

**解决方法**
```bash
pip install rl4co
```

### 问题 3: 登录后跳转回登录页

**解决方法**
1. 检查 `app.config['SECRET_KEY']` 是否设置
2. 清除浏览器 Cookie
3. 重启 Flask 服务器

### 问题 4: 图片不显示

**解决方法**
1. 确保 `static/model_plots/` 目录存在
2. 检查文件路径和权限
3. 查看浏览器控制台错误信息

### 问题 5: 训练速度慢

**解决方法**
1. 使用 GPU 训练（提升 10-50 倍）
2. 减小批次大小
3. 减少训练轮数
4. 使用更轻量的模型

---

## 📈 性能数据

### 训练速度 (TSP-50)

| 环境 | Batch Size | Speed (samples/s) |
|------|-----------|------------------|
| CPU (i7-12700) | 128 | ~50 |
| GPU (RTX 3060) | 512 | ~800 |
| GPU (RTX 4090) | 1024 | ~2000 |

### 模型性能 (TSP-50)

| 模型 | Avg Cost | Training Time |
|------|----------|--------------|
| Attention Model | 5.80 | 2h |
| POMO | 5.73 | 3h |
| SymNCO | 5.69 | 4h |

---

## 🚀 部署指南

### 开发环境

```bash
# 克隆项目
git clone https://github.com/your-repo/rl4co-display.git
cd rl4co-display

# 安装依赖
pip install -r requirements.txt

# 配置数据库
mysql -u root -p flaskdemo_user < database_init_with_auth.sql

# 启动服务
python app.py
```

### 生产环境

#### 使用 Gunicorn

```bash
# 安装 Gunicorn
pip install gunicorn

# 启动服务（4 个 worker）
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### 使用 Nginx 反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /static {
        alias /path/to/rl4co-display/static;
    }
}
```

#### 使用 Docker

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

### 报告问题

1. 检查是否已有相同问题
2. 提供详细的错误信息和复现步骤
3. 附上系统环境信息

### 提交代码

1. Fork 本项目
2. 创建新分支: `git checkout -b feature/your-feature`
3. 提交修改: `git commit -m "Add your feature"`
4. 推送分支: `git push origin feature/your-feature`
5. 提交 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [RL4CO](https://github.com/ai4co/rl4co) - 强化学习组合优化库
- [PyTorch Lightning](https://lightning.ai/) - 训练框架
- [Flask](https://flask.palletsprojects.com/) - Web 框架
- [ECharts](https://echarts.apache.org/) - 数据可视化库

---

## 📞 联系方式

- **项目主页**: [GitHub Repository]
- **文档**: [docs/README.md](docs/README.md)
- **问题反馈**: [GitHub Issues]

---

## 🎓 单位信息

**开发单位**: 山西大学 计算机科学与技术学院  
**项目用途**: 强化学习教学和科研  
**技术栈**: Python + Flask + PyTorch + RL4CO  

---

**版本**: 1.0.0  
**最后更新**: 2024  

---

**开始你的强化学习之旅！** 🚀


