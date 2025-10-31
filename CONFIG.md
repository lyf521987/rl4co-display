# RL4CO Display - 配置文件完整指南

> 山西大学 计算机科学与技术学院  
> 所有配置文件的详细说明和使用指南

---

## 📋 配置文件概览

本项目包含以下配置文件：

| 文件 | 说明 | 用途 |
|------|------|------|
| `config.py` | Python配置类 | 数据库连接配置 |
| `requirements.txt` | Python依赖列表 | 项目依赖管理 |
| `database_init_with_auth.sql` | 数据库初始化脚本 | 数据库表结构创建 |
| `.env` (可选) | 环境变量配置 | 敏感信息配置（不提交到版本控制） |

---

## 🔧 1. config.py - 应用配置

### 文件位置
```
rl4co-display/config.py
```

### 配置内容

```python
# config.py

class Config:
    # ========== 数据库配置 ==========
    MYSQL_HOST = 'localhost'          # 数据库主机地址
    MYSQL_USER = 'root'               # 数据库用户名
    MYSQL_PASSWORD = '2005'           # 数据库密码（⚠️ 请修改为您的密码）
    MYSQL_DB = 'flaskdemo_user'       # 数据库名称
```

### 配置说明

#### 数据库配置

- **MYSQL_HOST**: MySQL服务器地址
  - 本地开发: `'localhost'`
  - 远程服务器: `'192.168.1.100'` 或域名
  - Docker容器: `'mysql'` (容器名称)

- **MYSQL_USER**: MySQL用户名
  - 开发环境: `'root'`
  - 生产环境: 建议创建专用用户（如 `'flask_app'`）

- **MYSQL_PASSWORD**: MySQL密码
  - ⚠️ **重要**: 生产环境必须修改为强密码
  - 不要提交到版本控制系统

- **MYSQL_DB**: 数据库名称
  - 默认: `'flaskdemo_user'`
  - 可以根据需要修改

### 安全配置（在 app.py 中）

```python
# app.py 中的安全配置
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'rl4co-display-secret-key-2024-change-in-production'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
```

#### SECRET_KEY 生成方法

```bash
# 方法1: 使用 Python
python -c "import secrets; print(secrets.token_hex(32))"

# 方法2: 使用 OpenSSL
openssl rand -hex 32
```

### 环境变量配置（推荐）

创建 `.env` 文件（不要提交到版本控制）：

```bash
# .env
MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_password_here
MYSQL_DB=flaskdemo_user
SECRET_KEY=your-secret-key-here
```

使用 `python-dotenv` 加载：

```python
# config.py
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
    MYSQL_DB = os.environ.get('MYSQL_DB', 'flaskdemo_user')
```

---

## 📦 2. requirements.txt - Python依赖

### 文件位置
```
rl4co-display/requirements.txt
```

### 依赖列表

```txt
# Flask 核心依赖
Flask==3.0.0
flask-mysqldb==2.0.0
Werkzeug==3.0.1

# 强化学习相关依赖
torch>=2.0.0
rl4co>=0.4.0
lightning>=2.0.0

# 可视化依赖
matplotlib>=3.7.0
numpy>=1.24.0

# 其他依赖
python-dotenv==1.0.0
```

### 安装方法

```bash
# 标准安装
pip install -r requirements.txt

# 使用虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 生产环境额外依赖

```bash
# 如果需要使用 Gunicorn
pip install gunicorn

# 如果需要 Redis 缓存
pip install redis flask-caching
```

### 依赖版本说明

- **Flask 3.0.0**: Web框架核心
- **torch>=2.0.0**: PyTorch深度学习框架（支持CPU和GPU）
- **rl4co>=0.4.0**: 强化学习组合优化库
- **lightning>=2.0.0**: PyTorch Lightning训练框架
- **matplotlib>=3.7.0**: 数据可视化库

### CUDA支持（可选）

如果需要GPU加速，安装CUDA版本的PyTorch：

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🗄️ 3. database_init_with_auth.sql - 数据库初始化

### 文件位置
```
rl4co-display/database_init_with_auth.sql
```

### 使用方法

```bash
# 方法1: 命令行导入
mysql -u root -p flaskdemo_user < database_init_with_auth.sql

# 方法2: MySQL客户端
mysql -u root -p
source database_init_with_auth.sql;
```

### 数据库结构

#### 3.1 users 表 - 用户信息

```sql
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,      -- pbkdf2:sha256加密
    email VARCHAR(100),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP NULL,
    INDEX idx_username (username)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

**字段说明**:
- `id`: 用户ID（主键）
- `username`: 用户名（唯一）
- `password`: 密码哈希值（pbkdf2:sha256）
- `email`: 邮箱（可选）
- `create_time`: 创建时间
- `last_login`: 最后登录时间

#### 3.2 training_sessions 表 - 训练会话

```sql
CREATE TABLE training_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(50) UNIQUE NOT NULL,
    user_id INT NOT NULL,
    model_type VARCHAR(50) NOT NULL,        -- AM, POMO等
    problem_type VARCHAR(50) NOT NULL,      -- TSP, CVRP等
    config JSON,                            -- 训练配置参数
    status ENUM('running', 'completed', 'failed'),
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    final_loss DECIMAL(10, 4),
    final_reward DECIMAL(10, 4),
    best_reward DECIMAL(10, 4),
    checkpoint_path VARCHAR(255),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

#### 3.3 training_files 表 - 训练文件

```sql
CREATE TABLE training_files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    session_id VARCHAR(50) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type ENUM('plot', 'animation', 'curve', 'checkpoint'),
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 视图和存储过程

#### user_training_stats 视图

提供用户训练统计信息：

```sql
SELECT * FROM user_training_stats;
```

#### delete_user_and_data 存储过程

删除用户及其所有相关数据：

```sql
CALL delete_user_and_data(用户ID);
```

---

## 🔐 4. 安全配置最佳实践

### 4.1 生产环境配置清单

- [ ] 修改 `config.py` 中的数据库密码
- [ ] 生成并设置强 `SECRET_KEY`
- [ ] 创建专用的MySQL用户（非root）
- [ ] 限制MySQL用户权限
- [ ] 使用环境变量存储敏感信息
- [ ] 配置 `.gitignore` 排除 `.env` 文件
- [ ] 启用HTTPS（生产环境）
- [ ] 配置防火墙规则

### 4.2 MySQL用户权限配置

```sql
-- 创建应用专用用户
CREATE USER 'flask_app'@'localhost' IDENTIFIED BY 'strong_password_here';

-- 授予必要权限
GRANT SELECT, INSERT, UPDATE, DELETE ON flaskdemo_user.* TO 'flask_app'@'localhost';

-- 刷新权限
FLUSH PRIVILEGES;
```

### 4.3 环境变量示例

创建 `.env` 文件：

```bash
# 数据库配置
MYSQL_HOST=localhost
MYSQL_USER=flask_app
MYSQL_PASSWORD=your_secure_password_here
MYSQL_DB=flaskdemo_user

# Flask配置
SECRET_KEY=your-32-character-secret-key-here
FLASK_ENV=production
FLASK_DEBUG=False

# 其他配置（可选）
REDIS_URL=redis://localhost:6379/0
LOG_LEVEL=INFO
```

### 4.4 .gitignore 配置

确保 `.gitignore` 包含：

```gitignore
# 环境变量
.env
.env.local
.env.*.local

# 配置文件中的敏感信息
config.py
# 或者使用 config.py.example 作为模板
```

---

## 🚀 5. 快速配置指南

### 开发环境配置

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd rl4co-display
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **配置数据库**
   ```bash
   # 创建数据库
   mysql -u root -p
   CREATE DATABASE flaskdemo_user CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   
   # 初始化表结构
   mysql -u root -p flaskdemo_user < database_init_with_auth.sql
   ```

5. **修改配置**
   ```bash
   # 编辑 config.py
   # 修改 MYSQL_PASSWORD 为您的密码
   ```

6. **启动应用**
   ```bash
   python app.py
   ```

### 生产环境配置

参考 [DEPLOYMENT.md](DEPLOYMENT.md) 获取详细的部署指南。

---

## 🔍 6. 配置验证

### 检查数据库连接

```python
# test_db_connection.py
from config import Config
import mysql.connector

try:
    conn = mysql.connector.connect(
        host=Config.MYSQL_HOST,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD,
        database=Config.MYSQL_DB
    )
    print("✓ 数据库连接成功")
    conn.close()
except Exception as e:
    print(f"✗ 数据库连接失败: {e}")
```

### 检查Python依赖

```bash
# 检查所有依赖是否安装
pip check

# 列出已安装的包
pip list

# 验证关键包版本
python -c "import flask; print(f'Flask: {flask.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import rl4co; print(f'RL4CO: {rl4co.__version__}')"
```

### 检查数据库表

```sql
-- 连接到数据库
mysql -u root -p flaskdemo_user

-- 查看所有表
SHOW TABLES;

-- 检查表结构
DESCRIBE users;
DESCRIBE training_sessions;
DESCRIBE training_files;

-- 检查记录数
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM training_sessions;
SELECT COUNT(*) FROM training_files;
```

---

## 🐛 7. 常见配置问题

### 问题1: 数据库连接失败

**错误信息**:
```
✗ 数据库连接失败: Access denied for user 'root'@'localhost'
```

**解决方法**:
1. 检查 `config.py` 中的用户名和密码
2. 确认MySQL服务正在运行
3. 验证数据库是否存在

### 问题2: 模块导入错误

**错误信息**:
```
ModuleNotFoundError: No module named 'rl4co'
```

**解决方法**:
```bash
pip install -r requirements.txt
```

### 问题3: SECRET_KEY警告

**错误信息**:
```
警告: SECRET_KEY 未设置，使用默认值
```

**解决方法**:
- 设置环境变量 `SECRET_KEY`
- 或在 `app.py` 中修改默认值

### 问题4: 字符编码问题

**错误信息**:
```
UnicodeDecodeError: 'utf-8' codec can't decode
```

**解决方法**:
- 确保数据库使用 `utf8mb4` 字符集
- 检查 `database_init_with_auth.sql` 中的字符集设置

---

## 📝 8. 配置更新日志

### v1.0.0 (2024)
- ✅ 初始配置结构
- ✅ 数据库配置模块
- ✅ 安全配置支持
- ✅ 环境变量支持

---

## 📞 获取帮助

- **文档**: [README.md](README.md)
- **部署指南**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **详细文档**: [docs/](docs/)

---

**最后更新**: 2024年  
**项目**: RL4CO 强化学习优化平台  
**单位**: 山西大学 计算机科学与技术学院

