from flask import Flask, render_template, request, render_template_string, redirect, url_for, jsonify, Response
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from config import Config  # 导入配置文件
import json
import time
import threading
from queue import Queue
import torch
import os
import sys
from io import StringIO
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.pyplot as plt

# 导入 RL4CO 相关模块
try:
    from rl4co.envs import TSPEnv, CVRPEnv
    from rl4co.models import AttentionModelPolicy, REINFORCE
    from rl4co.utils.trainer import RL4COTrainer
    from lightning.pytorch.callbacks import Callback
    RL4CO_AVAILABLE = True
except ImportError:
    RL4CO_AVAILABLE = False
    print("警告: RL4CO 库未安装，将使用模拟训练模式")

app = Flask(__name__)
# 加载配置
app.config.from_object(Config)
mysql = MySQL(app)

# 用于存储训练状态和进度的全局字典
training_status = {}
training_queues = {}

# 创建输出目录
PLOTS_DIR = "static/model_plots"
CHECKPOINTS_DIR = "checkpoints"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

@app.route('/')
def Index_login():  # put application's code here
    return render_template('login.html')

@app.route('/res')
def Index_res():  # put application's code here
    return render_template('res.html')

# 注册路由
@app.route('/register', methods=['POST'])
def register():
    name = request.form.get('username')
    pwd = request.form.get('password')

    if not name or not pwd:
        return render_template_string("用户名和密码不能为空，<a href='/'>返回登录</a>"), 400

    # 检查用户名是否已经存在
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE username = %s", (name,))
    user = cur.fetchone()

    if user:
        return render_template_string("用户名已存在，<a href='/'>返回登录</a>"), 400

    # 将新用户添加到数据库
    hashed_pwd = generate_password_hash(pwd)
    cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (name, hashed_pwd))
    mysql.connection.commit()
    cur.close()

    return render_template_string("注册成功,<a href='/'>返回登录</a>"), 201


# 登录路由
@app.route('/login', methods=['POST'])
def login():
    name = request.form.get('username')
    pwd = request.form.get('password')

    if not name or not pwd:
        return render_template_string("用户名和密码不能为空，<a href='/'>返回注册</a>"), 400

        # 检查用户名和密码是否匹配
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE username = %s", (name,))
    user = cur.fetchone()

    if user and check_password_hash(user[2], pwd):  # user[2] 是密码字段
        # 登录成功，重定向到主页
        return redirect(url_for('home'))  # 'home' 是主页的路由函数名
    else:
        return "用户名或密码错误", 401

@app.route('/home')
def home():
    return render_template('index.html')  # 渲染主页模板


# 自定义 Lightning Callback 用于捕获训练进度
class ProgressCallback(Callback):  # 定义一个回调用于在训练过程中收集与推送指标
    def __init__(self, queue, session_id, total_epochs):  # 初始化回调实例
        super().__init__()  # 调用父类初始化
        self.queue = queue  # 保存与前端通信的消息队列
        self.session_id = session_id  # 保存当前训练会话ID
        self.total_epochs = total_epochs  # 保存总训练轮数，用于百分比计算
        self.best_reward = float('-inf')  # 记录历史最优奖励（越大越好）
        self.epoch_losses = []  # 存放当前epoch内每个batch的loss
        self.epoch_rewards = []  # 存放当前epoch内每个batch的reward
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # 每个batch结束时被调用
        """每个 batch 结束时收集指标"""  # 说明本函数用途：收集batch级指标
        # 尝试从多个来源获取 loss 和 reward  # 兼容不同版本/实现的输出结构
        loss_collected = False  # 标记是否已成功采集到loss
        reward_collected = False  # 标记是否已成功采集到reward
        
        # 方法1: 从 outputs 获取  # 首选从Lightning返回的outputs中读取
        if outputs is not None and isinstance(outputs, dict):  # 确认outputs为字典
            if 'loss' in outputs:  # 如果包含loss键
                loss_val = outputs['loss']  # 取出loss张量
                if isinstance(loss_val, torch.Tensor):  # 保证类型为Tensor
                    self.epoch_losses.append(loss_val.item())  # 转为标量并加入列表
                    loss_collected = True  # 标记已采集到loss
            
            if 'reward' in outputs:  # 如果包含reward键
                reward_val = outputs['reward']  # 取出reward张量（可能是batch维）
                if isinstance(reward_val, torch.Tensor):  # 保证类型为Tensor
                    self.epoch_rewards.append(reward_val.mean().item())  # 取均值转标量后加入列表
                    reward_collected = True  # 标记已采集到reward
        
        # 方法2: 从 pl_module 的 logged_metrics 获取  # 备选：从Lightning记录的指标中读取
        if not loss_collected and hasattr(pl_module, 'log_dict') and hasattr(trainer, 'logged_metrics'):  # 仅在未采集到loss时尝试
            logged = trainer.logged_metrics  # 读取已记录的指标字典
            if 'loss' in logged:  # 如果包含loss
                loss_val = logged['loss']  # 取出loss张量
                if isinstance(loss_val, torch.Tensor):  # 类型检查
                    self.epoch_losses.append(loss_val.item())  # 转为标量并记录
        
        if not reward_collected and hasattr(pl_module, 'log_dict') and hasattr(trainer, 'logged_metrics'):  # 未采集到reward时尝试
            logged = trainer.logged_metrics  # 读取记录的指标
            if 'reward' in logged:  # 如果包含reward
                reward_val = logged['reward']  # 取出reward张量
                if isinstance(reward_val, torch.Tensor):  # 类型检查
                    self.epoch_rewards.append(reward_val.item())  # 转为标量并记录
    
    def on_train_epoch_end(self, trainer, pl_module):  # 每个epoch结束时被调用
        """每个训练 epoch 结束时调用"""  # 说明本函数用途：汇总并推送epoch级指标
        epoch = trainer.current_epoch + 1  # 获取当前epoch编号，转为1起始
        
        # 首先尝试从累积的 batch 指标中计算平均值  # 以batch汇总的方式获得更稳定的统计
        loss = 0.0  # 初始化loss为0
        reward = 0.0  # 初始化reward为0
        
        if self.epoch_losses:  # 如果本epoch收集到了loss
            loss = sum(self.epoch_losses) / len(self.epoch_losses)  # 计算loss均值
        
        if self.epoch_rewards:  # 如果本epoch收集到了reward
            reward = sum(self.epoch_rewards) / len(self.epoch_rewards)  # 计算reward均值
        
        # 如果没有从 batch 中获取到，尝试从 metrics 获取  # 兼容某些情况下outputs未返回指标
        if loss == 0.0 or reward == 0.0:  # 只要有一个为0则尝试回退
            metrics = trainer.callback_metrics  # 从Lightning回调指标中读取
            
            # 调试：打印所有可用的指标键名（仅第一个epoch）  # 便于识别具体指标键
            if epoch == 1:  # 仅在首个epoch打印，避免刷屏
                self.queue.put(json.dumps({  # 通过队列向前端发送信息
                    'type': 'info',  # 消息类型为info
                    'message': f'可用的 callback_metrics 键: {list(metrics.keys())}'  # 列出callback_metrics键
                }))
                if hasattr(trainer, 'logged_metrics'):  # 如果存在logged_metrics
                    self.queue.put(json.dumps({  # 再发送一条消息
                        'type': 'info',  # 信息类型
                        'message': f'可用的 logged_metrics 键: {list(trainer.logged_metrics.keys())}'  # 列出logged_metrics键
                    }))
            
            # RL4CO 的 REINFORCE 模型使用的键名  # 依次尝试常见键名
            if loss == 0.0:  # 若loss仍未得到
                loss = metrics.get('loss', metrics.get('train_loss', metrics.get('train/loss', 0.0)))  # 多键名回退
            if reward == 0.0:  # 若reward仍未得到
                reward = metrics.get('reward', metrics.get('train_reward', metrics.get('train/reward', 0.0)))  # 多键名回退
            
            # 如果还是没有找到，尝试从 logged_metrics 获取  # 最后回退到logged_metrics
            if loss == 0.0 and hasattr(trainer, 'logged_metrics'):  # 若仍为0并且存在logged_metrics
                logged = trainer.logged_metrics  # 读取logged_metrics
                loss = logged.get('loss', logged.get('train_loss', logged.get('train/loss', 0.0)))  # 多键名回退
            
            if reward == 0.0 and hasattr(trainer, 'logged_metrics'):  # 若仍为0并且存在logged_metrics
                logged = trainer.logged_metrics  # 读取logged_metrics
                reward = logged.get('reward', logged.get('train_reward', logged.get('train/reward', 0.0)))  # 多键名回退
            
            if isinstance(loss, torch.Tensor):  # 如果loss还是张量
                loss = loss.item()  # 转为标量
            if isinstance(reward, torch.Tensor):  # 如果reward还是张量
                reward = reward.item()  # 转为标量
        
        # 清空本 epoch 的累积指标  # 为下一个epoch做准备
        self.epoch_losses = []  # 重置loss列表
        self.epoch_rewards = []  # 重置reward列表
        
        self.best_reward = max(self.best_reward, reward)  # 更新历史最优reward
        progress = (epoch / self.total_epochs) * 100  # 计算训练进度百分比
        
        # 更新训练状态  # 将最新指标写入全局状态，供查询接口使用
        training_status[self.session_id].update({
            'progress': progress,  # 当前进度百分比
            'epoch': epoch,  # 当前epoch编号
            'loss': round(loss, 4),  # 本epoch平均loss（四舍五入）
            'reward': round(reward, 4),  # 本epoch平均reward（四舍五入）
            'best_reward': round(self.best_reward, 4)  # 历史最优reward（四舍五入）
        })
        
        # 发送进度更新  # 以SSE消息形式推送进度到前端
        self.queue.put(json.dumps({
            'type': 'progress',  # 消息类型：进度
            'epoch': epoch,  # 当前epoch
            'total_epochs': self.total_epochs,  # 总epoch数
            'progress': round(progress, 2),  # 进度百分比保留两位
            'loss': round(loss, 4),  # 平均loss
            'reward': round(reward, 4),  # 平均reward
            'best_reward': round(self.best_reward, 4)  # 历史最优reward
        }))
        
        # 发送详细信息  # 额外以info形式发送可读字符串
        self.queue.put(json.dumps({
            'type': 'info',  # 消息类型：信息
            'message': f'Epoch {epoch}/{self.total_epochs} - Loss: {loss:.4f}, Reward: {reward:.4f}, Best: {self.best_reward:.4f}'  # 格式化的训练摘要
        }))


# 真实的 RL4CO 训练函数
def real_rl4co_training(config, session_id):  # 使用RL4CO执行真实训练流程
    """使用 RL4CO 进行真实的强化学习训练"""  # 函数说明：真实训练模式
    queue = training_queues[session_id]  # 取出当前会话的消息队列
    
    try:  # 捕获训练过程中的异常
        # 初始化训练状态  # 为前端展示准备默认状态
        training_status[session_id] = {
            'status': 'running',  # 标记状态为运行中
            'progress': 0,  # 初始进度0
            'epoch': 0,  # 当前epoch为0
            'loss': 0,  # 初始loss为0
            'reward': 0,  # 初始reward为0
            'best_reward': 0  # 初始best为0
        }
        
        # 获取配置参数  # 从请求配置中解析训练超参
        epochs = int(config.get('epochs', 3))  # 训练轮数，默认3
        model_type = config.get('model', 'attention')  # 模型类型，默认attention
        problem_type = config.get('problem', 'tsp')  # 问题类型，默认tsp
        batch_size = int(config.get('batch_size', 512))  # batch大小，默认512
        learning_rate = float(config.get('learning_rate', 1e-4))  # 学习率，默认1e-4
        num_loc = 50  # 问题规模（TSP点数）
        
        # 发送训练开始消息  # 告知前端训练已启动及配置信息
        queue.put(json.dumps({
            'type': 'info',  # 消息类型
            'message': f'开始训练 {model_type.upper()} 模型，问题类型: {problem_type.upper()}'  # 文本内容
        }))
        
        queue.put(json.dumps({
            'type': 'info',  # 消息类型
            'message': f'配置: Epochs={epochs}, Batch={batch_size}, LR={learning_rate}, 问题规模={num_loc}'  # 配置详情
        }))
        
        # 检测设备  # 自动选择GPU或CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch设备
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"  # Lightning加速器类型
        devices = 1 if torch.cuda.is_available() else "auto"  # 设备数量设置
        
        queue.put(json.dumps({
            'type': 'info',  # 消息类型
            'message': f'使用设备: {device}'  # 展示设备信息
        }))
        
        # 初始化环境  # 根据问题类型构造环境
        if problem_type.lower() == 'tsp':  # TSP问题
            env = TSPEnv(generator_params={'num_loc': num_loc})  # 创建TSP环境
        elif problem_type.lower() == 'cvrp':  # CVRP问题
            env = CVRPEnv(generator_params={'num_loc': num_loc})  # 创建CVRP环境
        else:  # 其他情况默认TSP
            env = TSPEnv(generator_params={'num_loc': num_loc})  # 创建默认TSP环境
        
        queue.put(json.dumps({
            'type': 'info',  # 消息类型
            'message': f'环境初始化完成: {env.name}'  # 环境名称反馈
        }))
        
        # 定义策略网络  # 构建注意力模型策略
        policy = AttentionModelPolicy(
            env_name=env.name,  # 指定环境名，以匹配输入输出
            embed_dim=128,  # 嵌入维度
            num_encoder_layers=3,  # 编码器层数
            num_heads=8,  # 多头注意力的头数
        )
        
        queue.put(json.dumps({
            'type': 'info',  # 消息类型
            'message': '策略网络初始化完成'  # 策略构建完成
        }))
        
        # 定义 RL 模型  # 使用REINFORCE策略梯度
        model = REINFORCE(
            env,  # 传入环境
            policy,  # 传入策略网络
            baseline="rollout",  # 基线类型，采用rollout baseline
            batch_size=batch_size,  # 训练batch大小
            train_data_size=10_000,  # 减少训练数据量以提升速度
            val_data_size=1_000,  # 验证数据量
            optimizer_kwargs={"lr": learning_rate},  # 优化器超参
        )
        
        # 检查是否有已保存的 checkpoint  # 支持断点续训
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{problem_type}-{model_type}.ckpt")  # 目标ckpt路径
        ckpt_path = checkpoint_path if os.path.exists(checkpoint_path) else None  # 若存在则使用
        
        if ckpt_path:  # 如果找到了历史ckpt
            queue.put(json.dumps({
                'type': 'info',  # 消息类型
                'message': f'加载检查点: {checkpoint_path}'  # 提示已加载
            }))
        
        # 创建进度回调  # 构建自定义回调以推送指标
        progress_callback = ProgressCallback(queue, session_id, epochs)  # 实例化回调
        
        # 初始化训练器  # 构建Lightning训练器
        trainer = RL4COTrainer(
            max_epochs=epochs,  # 最大全训练轮数
            accelerator=accelerator,  # 加速器设置
            devices=devices,  # 设备配置
            callbacks=[progress_callback],  # 注册回调
            logger=None,  # 关闭默认日志记录器
            enable_progress_bar=False,  # 关闭进度条
            enable_model_summary=False,  # 关闭模型摘要
        )
        
        queue.put(json.dumps({
            'type': 'info',  # 消息类型
            'message': '开始训练...'  # 提示训练开始
        }))
        
        # 开始训练  # 执行fit，支持从ckpt继续
        if ckpt_path:  # 若存在ckpt则从ckpt继续
            trainer.fit(model, ckpt_path=ckpt_path)  # 带ckpt训练
        else:  # 否则从头训练
            trainer.fit(model)  # 直接训练
        
        queue.put(json.dumps({
            'type': 'info',  # 消息类型
            'message': '训练完成，开始生成可视化结果...'  # 训练结束提示
        }))
        
        # 训练后测试并生成可视化  # 对比随机策略与训练后策略
        policy = model.policy.to(device)  # 将策略移动到目标设备
        td_init = env.reset(batch_size=[3]).to(device)  # 生成3个测试实例并放到设备
        
        # 未训练模型测试（使用随机策略）  # 采样解码模拟未训练表现
        out_untrained = policy(td_init.clone(), phase="test", decode_type="sampling", return_actions=True)  # 前向计算
        actions_untrained = out_untrained['actions'].cpu().detach()  # 提取动作并转CPU
        rewards_untrained = out_untrained['reward'].cpu().detach()  # 提取奖励并转CPU
        
        # 训练后模型测试  # 贪心解码评估训练后性能
        out_trained = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)  # 前向计算
        actions_trained = out_trained['actions'].cpu().detach()  # 提取动作并转CPU
        rewards_trained = out_trained['reward'].cpu().detach()  # 提取奖励并转CPU
        
        # 生成对比图  # 可视化随机与训练后路径及代价
        plot_paths = []  # 存储生成图片的相对路径
        for i, td in enumerate(td_init):  # 遍历每个测试实例
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 创建左右两个子图
            env.render(td, actions_untrained[i], ax=axs[0])  # 左图渲染随机策略路径
            env.render(td, actions_trained[i], ax=axs[1])  # 右图渲染训练后策略路径
            axs[0].set_title(f"Random | Cost = {-rewards_untrained[i].item():.3f}")  # 左图标题：随机策略成本
            axs[1].set_title(f"Trained | Cost = {-rewards_trained[i].item():.3f}")  # 右图标题：训练后成本
            
            plot_filename = f"comparison_{session_id[:8]}_{i+1}.png"  # 生成图片文件名（含会话前缀）
            plot_path = os.path.join(PLOTS_DIR, plot_filename)  # 拼接完整保存路径
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")  # 保存图片到磁盘
            plt.close()  # 关闭图像以释放内存
            plot_paths.append(f"/static/model_plots/{plot_filename}")  # 记录供前端展示的路径
        
        # 保存检查点  # 将最终模型权重保存到文件
        trainer.save_checkpoint(checkpoint_path)  # 保存ckpt
        
        queue.put(json.dumps({
            'type': 'info',  # 消息类型
            'message': f'检查点已保存: {checkpoint_path}'  # 保存成功提示
        }))
        
        # 训练完成  # 汇总最终结果并通知前端
        training_status[session_id]['status'] = 'completed'  # 标记状态为已完成
        final_results = {
            'model': model_type,  # 模型类型
            'problem': problem_type,  # 问题类型
            'strategy': 'REINFORCE',  # 训练策略
            'total_epochs': epochs,  # 总训练轮数
            'final_loss': training_status[session_id]['loss'],  # 最终loss
            'final_reward': training_status[session_id]['reward'],  # 最终reward
            'best_reward': training_status[session_id]['best_reward'],  # 历史最优reward
            'plot_paths': plot_paths,  # 可视化图片路径
            'checkpoint_path': checkpoint_path  # 模型ckpt路径
        }
        
        queue.put(json.dumps({
            'type': 'complete',  # 消息类型：完成
            'message': '训练完成！',  # 完成提示
            'results': final_results  # 附带最终结果数据
        }))
        
    except Exception as e:  # 异常处理分支
        import traceback  # 引入traceback用于堆栈信息
        error_msg = f'{str(e)}\n{traceback.format_exc()}'  # 组装错误与堆栈文本（便于调试）
        training_status[session_id]['status'] = 'error'  # 将状态置为错误
        queue.put(json.dumps({  # 向前端推送错误消息
            'type': 'error',  # 消息类型：错误
            'message': f'训练出错: {str(e)}'  # 错误描述
        }))


# 模拟训练函数（备用）
def simulate_training(config, session_id):
    """模拟强化学习训练过程（当 RL4CO 不可用时）"""
    queue = training_queues[session_id]
    
    try:
        training_status[session_id] = {
            'status': 'running',
            'progress': 0,
            'epoch': 0,
            'loss': 0,
            'reward': 0,
            'best_reward': 0
        }
        
        epochs = int(config.get('epochs', 10))
        model = config.get('model', 'attention')
        problem = config.get('problem', 'tsp')
        
        queue.put(json.dumps({
            'type': 'info',
            'message': f'[模拟模式] 开始训练 {model.upper()} 模型，问题类型: {problem.upper()}'
        }))
        
        for epoch in range(1, epochs + 1):
            time.sleep(0.5)
            
            import random
            progress = (epoch / epochs) * 100
            loss = 10 * (1 - epoch / epochs) + random.uniform(0, 0.5)
            reward = -20 + (15 * epoch / epochs) + random.uniform(-1, 1)
            best_reward = max(training_status[session_id].get('best_reward', float('-inf')), reward)
            
            training_status[session_id].update({
                'progress': progress,
                'epoch': epoch,
                'loss': round(loss, 4),
                'reward': round(reward, 4),
                'best_reward': round(best_reward, 4)
            })
            
            queue.put(json.dumps({
                'type': 'progress',
                'epoch': epoch,
                'total_epochs': epochs,
                'progress': round(progress, 2),
                'loss': round(loss, 4),
                'reward': round(reward, 4),
                'best_reward': round(best_reward, 4)
            }))
            
            if epoch % 2 == 0 or epoch == epochs:
                queue.put(json.dumps({
                    'type': 'info',
                    'message': f'Epoch {epoch}/{epochs} - Loss: {loss:.4f}, Reward: {reward:.4f}'
                }))
        
        training_status[session_id]['status'] = 'completed'
        queue.put(json.dumps({
            'type': 'complete',
            'message': '[模拟模式] 训练完成！',
            'results': {
                'model': model,
                'problem': problem,
                'strategy': 'REINFORCE',
                'total_epochs': epochs,
                'final_loss': training_status[session_id]['loss'],
                'final_reward': training_status[session_id]['reward'],
                'best_reward': training_status[session_id]['best_reward']
            }
        }))
        
    except Exception as e:
        training_status[session_id]['status'] = 'error'
        queue.put(json.dumps({
            'type': 'error',
            'message': f'训练出错: {str(e)}'
        }))


@app.route('/api/start_training', methods=['POST'])
def start_training():
    """接收训练配置并启动训练"""
    try:
        config = request.json
        
        # 生成唯一的会话 ID
        import uuid
        session_id = str(uuid.uuid4())
        
        # 创建消息队列
        training_queues[session_id] = Queue()
        
        # 根据 RL4CO 是否可用选择训练函数
        if RL4CO_AVAILABLE:
            training_func = real_rl4co_training
            mode = "真实训练模式"
        else:
            training_func = simulate_training
            mode = "模拟训练模式"
        
        # 在后台线程中启动训练
        training_thread = threading.Thread(
            target=training_func,
            args=(config, session_id),
            daemon=True
        )
        training_thread.start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': f'训练已启动 ({mode})'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'启动训练失败: {str(e)}'
        }), 500


@app.route('/api/training_progress/<session_id>')
def training_progress(session_id):
    """使用 Server-Sent Events (SSE) 推送训练进度"""
    def generate():
        if session_id not in training_queues:
            yield f"data: {json.dumps({'type': 'error', 'message': '无效的会话 ID'})}\n\n"
            return
        
        queue = training_queues[session_id]
        
        while True:
            try:
                # 从队列中获取消息（阻塞等待）
                message = queue.get(timeout=1)
                yield f"data: {message}\n\n"
                
                # 如果收到完成或错误消息，则结束流
                data = json.loads(message)
                if data['type'] in ['complete', 'error']:
                    break
                    
            except:
                # 超时或队列为空，发送心跳
                if session_id in training_status:
                    if training_status[session_id]['status'] == 'completed':
                        break
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/training_status/<session_id>')
def get_training_status(session_id):
    """获取当前训练状态"""
    if session_id in training_status:
        return jsonify({
            'success': True,
            'status': training_status[session_id]
        })
    else:
        return jsonify({
            'success': False,
            'message': '未找到训练会话'
        }), 404


if __name__ == '__main__':
    app.run(debug=True)
