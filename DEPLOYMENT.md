# 部署指南

> RL4CO Display - 快速部署到生产环境

---

## 🎯 快速导航：选择适合你的部署方式

本项目支持三种主要部署方式，选择一种最适合你的方案：

| 部署方式 | 难度 | 适用场景 | 优点 |
|---------|------|---------|------|
| **方法 1: Gunicorn + Nginx** | ⭐⭐⭐ | Linux 服务器 | 稳定、性能好、可靠性高 |
| **方法 2: Systemd 服务** | ⭐⭐ | Linux 服务器 | 开机自启、管理方便 |
| **方法 3: Docker** | ⭐⭐ | 云平台/容器化 | 快速、隔离、易于扩展 |

> **推荐方案**: 
> - 如果有 Linux 服务器 → **Systemd 服务** (最简单) 或 **Gunicorn + Nginx** (最稳定)
> - 如果使用云平台（阿里云、腾讯云等） → **Docker** (最方便)
> - 如果在 Windows 上 → **Docker** 或直接运行 `python app.py`

---

## 📋 部署前检查清单

- [ ] Python 3.8+ 已安装
- [ ] MySQL 8.0+ 已安装并运行
- [ ] 服务器环境已配置（如使用云服务器）
- [ ] 域名已配置（可选）
- [ ] SSL 证书已准备（生产环境推荐）

---

## 🚀 快速部署（5 步完成）

### 步骤 1: 下载项目

```bash
git clone https://github.com/your-repo/rl4co-display.git
cd rl4co-display
```

### 步骤 2: 安装依赖

```bash
pip install -r requirements.txt
```

**生产环境额外依赖**：
```bash
pip install gunicorn
```

### 步骤 3: 配置数据库

**3.1 创建数据库**

```bash
mysql -u root -p
```

```sql
CREATE DATABASE flaskdemo_user CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
EXIT;
```

**3.2 初始化数据表**

```bash
mysql -u root -p flaskdemo_user < database_init_with_auth.sql
```

**3.3 配置数据库连接**

编辑 `config.py`：

```python
# 数据库配置
MYSQL_HOST = 'localhost'           # 数据库主机
MYSQL_USER = 'root'                # 数据库用户名
MYSQL_PASSWORD = 'your_password'   # 数据库密码
MYSQL_DB = 'flaskdemo_user'        # 数据库名称

# 安全密钥（请修改为随机字符串）
SECRET_KEY = 'your-random-secret-key-here-change-me'
```

**生成安全的 SECRET_KEY**：
```python
python -c "import secrets; print(secrets.token_hex(32))"
```

### 步骤 4: 测试运行

```bash
python app.py
```

访问 `http://localhost:5000` 确认系统正常运行。

### 步骤 5: 生产环境部署

#### 方法 1: 使用 Gunicorn

```bash
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 300 app:app
```

参数说明：
- `-w 4`: 4 个 worker 进程
- `-b 0.0.0.0:5000`: 绑定所有网络接口的 5000 端口
- `--timeout 300`: 超时时间 300 秒（训练任务可能较长）

#### 方法 2: 使用 Systemd 服务（推荐）

创建服务文件 `/etc/systemd/system/rl4co-display.service`：

```ini
[Unit]
Description=RL4CO Display Web Application
After=network.target mysql.service

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/rl4co-display
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 --timeout 300 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl start rl4co-display
sudo systemctl enable rl4co-display  # 开机自启
```

查看状态：

```bash
sudo systemctl status rl4co-display
```

#### 方法 3: 使用 Docker

**创建 Dockerfile**（已包含在项目中）：

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p static/model_plots checkpoints

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "--timeout", "300", "app:app"]
```

**构建并运行**：

```bash
docker build -t rl4co-display .
docker run -d -p 5000:5000 --name rl4co-display rl4co-display
```

---

## 🔧 Nginx 反向代理配置

创建配置文件 `/etc/nginx/sites-available/rl4co-display`：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 如果使用 HTTPS，添加以下配置
    # listen 443 ssl;
    # ssl_certificate /path/to/cert.pem;
    # ssl_certificate_key /path/to/key.pem;

    client_max_body_size 100M;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # SSE 支持（实时训练曲线需要）
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }

    location /static {
        alias /path/to/rl4co-display/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

启用配置：

```bash
sudo ln -s /etc/nginx/sites-available/rl4co-display /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## 🔐 安全配置

### 1. 修改默认配置

**必须修改**：
- `config.py` 中的 `SECRET_KEY`
- 数据库密码
- 删除或修改测试账户

### 2. 配置防火墙

```bash
# 允许 HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# 不要直接暴露 5000 端口到外网
sudo ufw deny 5000/tcp
```

### 3. 数据库安全

```bash
# 运行 MySQL 安全配置脚本
mysql_secure_installation
```

确保：
- 设置强密码
- 删除测试数据库
- 禁用远程 root 登录

### 4. 定期备份

**数据库备份**：

```bash
# 创建备份脚本
cat > /usr/local/bin/backup-rl4co.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backup/rl4co-display"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p $BACKUP_DIR

# 备份数据库
mysqldump -u root -p flaskdemo_user > $BACKUP_DIR/db_$DATE.sql

# 备份文件
tar -czf $BACKUP_DIR/files_$DATE.tar.gz \
    /path/to/rl4co-display/static/model_plots \
    /path/to/rl4co-display/checkpoints

# 删除 7 天前的备份
find $BACKUP_DIR -type f -mtime +7 -delete
EOF

chmod +x /usr/local/bin/backup-rl4co.sh
```

**添加定时任务**：

```bash
crontab -e
```

添加：
```
0 2 * * * /usr/local/bin/backup-rl4co.sh
```

---

## 📊 性能优化

### 1. 使用多个 Worker

根据 CPU 核心数调整 Gunicorn worker 数量：

```bash
gunicorn -w $((2 * $(nproc) + 1)) -b 0.0.0.0:5000 app:app
```

### 2. 配置静态文件缓存

在 Nginx 中配置：

```nginx
location /static {
    alias /path/to/rl4co-display/static;
    expires 30d;
    add_header Cache-Control "public, immutable";
    gzip on;
    gzip_types text/css application/javascript image/svg+xml;
}
```

### 3. 使用 Redis 缓存（可选）

```bash
pip install redis flask-caching
```

在 `app.py` 中配置：

```python
from flask_caching import Cache

cache = Cache(app, config={
    'CACHE_TYPE': 'redis',
    'CACHE_REDIS_URL': 'redis://localhost:6379/0'
})
```

### 4. GPU 加速

如果服务器有 GPU：

```bash
# 安装 CUDA 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🐛 故障排查

### 问题 1: 无法连接数据库

**检查**：
```bash
sudo systemctl status mysql
mysql -u root -p
```

**解决**：
- 确保 MySQL 正在运行
- 检查 `config.py` 中的配置
- 检查防火墙设置

### 问题 2: Gunicorn Worker 超时

**症状**：训练任务运行时 worker 重启

**解决**：增加超时时间
```bash
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 600 app:app
```

### 问题 3: 静态文件 404

**检查**：
- 确认 `static/` 目录存在
- 检查文件权限
- 查看 Nginx 配置中的路径

### 问题 4: 内存不足

**查看内存使用**：
```bash
free -h
htop
```

**解决**：
- 减少 worker 数量
- 使用 swap
- 升级服务器配置

### 问题 5: 训练速度慢

**优化**：
- 使用 GPU
- 减小批次大小
- 使用更轻量的模型

---

## 📈 监控和日志

### 1. 应用日志

**查看实时日志**：

```bash
# Systemd 服务
sudo journalctl -u rl4co-display -f

# Docker
docker logs -f rl4co-display
```

**日志轮转**（/etc/logrotate.d/rl4co-display）：

```
/var/log/rl4co-display/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

### 2. 性能监控

```bash
# 安装监控工具
pip install flask-monitoring-dashboard

# 或使用系统监控
htop
iotop
nethogs
```

### 3. 错误告警

配置邮件或 Slack 通知（可选）。

---

## 🔄 更新和维护

### 更新应用

```bash
cd /path/to/rl4co-display
git pull
pip install -r requirements.txt --upgrade
sudo systemctl restart rl4co-display
```

### 数据库迁移

如果有数据库结构变更：

```bash
# 备份
mysqldump -u root -p flaskdemo_user > backup.sql

# 执行迁移脚本
mysql -u root -p flaskdemo_user < migration.sql
```

### 清理旧文件

```bash
# 清理超过 30 天的训练文件
find static/model_plots -type f -mtime +30 -delete

# 清理旧的检查点
find checkpoints -type f -mtime +30 -delete
```

---

## ✅ 部署验证

部署完成后，验证以下功能：

- [ ] 访问主页正常
- [ ] 用户注册功能正常
- [ ] 用户登录功能正常
- [ ] 开始训练功能正常
- [ ] 实时曲线更新正常
- [ ] 文件管理功能正常
- [ ] 算法对比页面正常
- [ ] 模型知识库页面正常

**运行自动化测试**：

```bash
python test_auth_功能测试.py
```

---

## 📞 技术支持

- **文档**: [README.md](README.md)
- **详细文档**: [docs/](docs/)
- **问题反馈**: GitHub Issues

---

## 📄 许可证

本项目采用 MIT 许可证

---

**祝部署顺利！** 🚀

如有问题，请参考主文档或提交 Issue。


