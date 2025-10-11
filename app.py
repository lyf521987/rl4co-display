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
class ProgressCallback(Callback):
    def __init__(self, queue, session_id, total_epochs):
        super().__init__()
        self.queue = queue
        self.session_id = session_id
        self.total_epochs = total_epochs
        self.best_reward = float('-inf')
    
    def on_train_epoch_end(self, trainer, pl_module):
        """每个训练 epoch 结束时调用"""
        epoch = trainer.current_epoch + 1
        
        # 获取训练指标
        metrics = trainer.callback_metrics
        loss = metrics.get('train_loss', 0.0)
        reward = metrics.get('train_reward', 0.0)
        
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if isinstance(reward, torch.Tensor):
            reward = reward.item()
        
        self.best_reward = max(self.best_reward, reward)
        progress = (epoch / self.total_epochs) * 100
        
        # 更新训练状态
        training_status[self.session_id].update({
            'progress': progress,
            'epoch': epoch,
            'loss': round(loss, 4),
            'reward': round(reward, 4),
            'best_reward': round(self.best_reward, 4)
        })
        
        # 发送进度更新
        self.queue.put(json.dumps({
            'type': 'progress',
            'epoch': epoch,
            'total_epochs': self.total_epochs,
            'progress': round(progress, 2),
            'loss': round(loss, 4),
            'reward': round(reward, 4),
            'best_reward': round(self.best_reward, 4)
        }))
        
        # 发送详细信息
        self.queue.put(json.dumps({
            'type': 'info',
            'message': f'Epoch {epoch}/{self.total_epochs} - Loss: {loss:.4f}, Reward: {reward:.4f}, Best: {self.best_reward:.4f}'
        }))


# 真实的 RL4CO 训练函数
def real_rl4co_training(config, session_id):
    """使用 RL4CO 进行真实的强化学习训练"""
    queue = training_queues[session_id]
    
    try:
        # 初始化训练状态
        training_status[session_id] = {
            'status': 'running',
            'progress': 0,
            'epoch': 0,
            'loss': 0,
            'reward': 0,
            'best_reward': 0
        }
        
        # 获取配置参数
        epochs = int(config.get('epochs', 3))
        model_type = config.get('model', 'attention')
        problem_type = config.get('problem', 'tsp')
        batch_size = int(config.get('batch_size', 512))
        learning_rate = float(config.get('learning_rate', 1e-4))
        num_loc = 50  # 问题规模
        
        # 发送训练开始消息
        queue.put(json.dumps({
            'type': 'info',
            'message': f'开始训练 {model_type.upper()} 模型，问题类型: {problem_type.upper()}'
        }))
        
        queue.put(json.dumps({
            'type': 'info',
            'message': f'配置: Epochs={epochs}, Batch={batch_size}, LR={learning_rate}, 问题规模={num_loc}'
        }))
        
        # 检测设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        devices = 1 if torch.cuda.is_available() else "auto"
        
        queue.put(json.dumps({
            'type': 'info',
            'message': f'使用设备: {device}'
        }))
        
        # 初始化环境
        if problem_type.lower() == 'tsp':
            env = TSPEnv(generator_params={'num_loc': num_loc})
        elif problem_type.lower() == 'cvrp':
            env = CVRPEnv(generator_params={'num_loc': num_loc})
        else:
            env = TSPEnv(generator_params={'num_loc': num_loc})
        
        queue.put(json.dumps({
            'type': 'info',
            'message': f'环境初始化完成: {env.name}'
        }))
        
        # 定义策略网络
        policy = AttentionModelPolicy(
            env_name=env.name,
            embed_dim=128,
            num_encoder_layers=3,
            num_heads=8,
        )
        
        queue.put(json.dumps({
            'type': 'info',
            'message': '策略网络初始化完成'
        }))
        
        # 定义 RL 模型
        model = REINFORCE(
            env,
            policy,
            baseline="rollout",
            batch_size=batch_size,
            train_data_size=10_000,  # 减少数据量以加快训练
            val_data_size=1_000,
            optimizer_kwargs={"lr": learning_rate},
        )
        
        # 检查是否有已保存的 checkpoint
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"{problem_type}-{model_type}.ckpt")
        ckpt_path = checkpoint_path if os.path.exists(checkpoint_path) else None
        
        if ckpt_path:
            queue.put(json.dumps({
                'type': 'info',
                'message': f'加载检查点: {checkpoint_path}'
            }))
        
        # 创建进度回调
        progress_callback = ProgressCallback(queue, session_id, epochs)
        
        # 初始化训练器
        trainer = RL4COTrainer(
            max_epochs=epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=[progress_callback],
            logger=None,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        
        queue.put(json.dumps({
            'type': 'info',
            'message': '开始训练...'
        }))
        
        # 开始训练
        if ckpt_path:
            trainer.fit(model, ckpt_path=ckpt_path)
        else:
            trainer.fit(model)
        
        queue.put(json.dumps({
            'type': 'info',
            'message': '训练完成，开始生成可视化结果...'
        }))
        
        # 训练后测试并生成可视化
        policy = model.policy.to(device)
        td_init = env.reset(batch_size=[3]).to(device)
        
        # 未训练模型测试（使用随机策略）
        out_untrained = policy(td_init.clone(), phase="test", decode_type="sampling", return_actions=True)
        actions_untrained = out_untrained['actions'].cpu().detach()
        rewards_untrained = out_untrained['reward'].cpu().detach()
        
        # 训练后模型测试
        out_trained = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
        actions_trained = out_trained['actions'].cpu().detach()
        rewards_trained = out_trained['reward'].cpu().detach()
        
        # 生成对比图
        plot_paths = []
        for i, td in enumerate(td_init):
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            env.render(td, actions_untrained[i], ax=axs[0])
            env.render(td, actions_trained[i], ax=axs[1])
            axs[0].set_title(f"Random | Cost = {-rewards_untrained[i].item():.3f}")
            axs[1].set_title(f"Trained | Cost = {-rewards_trained[i].item():.3f}")
            
            plot_filename = f"comparison_{session_id[:8]}_{i+1}.png"
            plot_path = os.path.join(PLOTS_DIR, plot_filename)
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            plot_paths.append(f"/static/model_plots/{plot_filename}")
        
        # 保存检查点
        trainer.save_checkpoint(checkpoint_path)
        
        queue.put(json.dumps({
            'type': 'info',
            'message': f'检查点已保存: {checkpoint_path}'
        }))
        
        # 训练完成
        training_status[session_id]['status'] = 'completed'
        final_results = {
            'model': model_type,
            'problem': problem_type,
            'strategy': 'REINFORCE',
            'total_epochs': epochs,
            'final_loss': training_status[session_id]['loss'],
            'final_reward': training_status[session_id]['reward'],
            'best_reward': training_status[session_id]['best_reward'],
            'plot_paths': plot_paths,
            'checkpoint_path': checkpoint_path
        }
        
        queue.put(json.dumps({
            'type': 'complete',
            'message': '训练完成！',
            'results': final_results
        }))
        
    except Exception as e:
        import traceback
        error_msg = f'{str(e)}\n{traceback.format_exc()}'
        training_status[session_id]['status'] = 'error'
        queue.put(json.dumps({
            'type': 'error',
            'message': f'训练出错: {str(e)}'
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
