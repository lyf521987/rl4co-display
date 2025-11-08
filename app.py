from flask import Flask, render_template, request, render_template_string, redirect, url_for, jsonify, Response, session, g
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
from PIL import Image
import numpy as np
from datetime import timedelta, datetime
from functools import wraps
import uuid as uuid_module
import mysql.connector as mysql_connector

# ========== 导入认证模块 ==========
from auth_module import (
    login_required, 
    UserManager, 
    TrainingSessionManager,
    FileManager,
    get_user_plot_dir,
    get_user_checkpoint_dir,
    set_user_session,
    clear_user_session,
    get_current_user_id,
    get_current_username,
    safe_join_path
)

# 配置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

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

# ========== 添加 SECRET_KEY 配置（用户认证必需） ==========
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or 'rl4co-display-secret-key-2024-change-in-production'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

mysql = MySQL(app)

# ========== 数据库连接管理（使用 Flask 请求上下文） ==========
def get_db():
    """获取当前请求的数据库连接（线程安全）"""
    if 'db' not in g:
        try:
            g.db = mysql_connector.connect(
                host=Config.MYSQL_HOST,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                autocommit=True
            )
        except Exception as e:
            print(f"✗ 数据库连接失败: {str(e)}")
            g.db = None
    return g.db

@app.teardown_appcontext
def close_db(error):
    """在请求结束时关闭数据库连接"""
    db = g.pop('db', None)
    if db is not None:
        try:
            db.close()
        except:
            pass

def get_user_manager():
    """获取当前请求的 UserManager 实例"""
    db = get_db()
    if db is None:
        return None
    if 'user_manager' not in g:
        g.user_manager = UserManager(db)
    return g.user_manager

def get_session_manager():
    """获取当前请求的 TrainingSessionManager 实例"""
    db = get_db()
    if db is None:
        return None
    if 'session_manager' not in g:
        g.session_manager = TrainingSessionManager(db)
    return g.session_manager

def get_file_manager():
    """获取当前请求的 FileManager 实例"""
    db = get_db()
    if db is None:
        return None
    if 'file_manager' not in g:
        g.file_manager = FileManager(db)
    return g.file_manager

def get_background_db():
    """为后台任务创建独立的数据库连接（不使用 Flask g 对象）"""
    try:
        db = mysql_connector.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            database=Config.MYSQL_DB,
            autocommit=True
        )
        return db
    except Exception as e:
        print(f"✗ 后台数据库连接失败: {str(e)}")
        return None

print("✓ 用户认证模块（请求上下文模式）配置完成")

# 用于存储训练状态和进度的全局字典
training_status = {}
training_queues = {}

# 创建输出目录
PLOTS_DIR = "static/model_plots"
CHECKPOINTS_DIR = "checkpoints"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# ========== 性能优化：API响应缓存 ==========
class SimpleCache:
    """简单的API响应缓存"""
    def __init__(self, timeout=300):
        self.cache = {}
        self.timeout = timeout
    
    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.timeout:
                return data
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = (value, time.time())
    
    def clear(self):
        self.cache.clear()

api_cache = SimpleCache(timeout=300)  # 5分钟缓存

def cached_api(key_prefix=''):
    """API缓存装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # 生成缓存键
            user_id = get_current_user_id()
            cache_key = f"{key_prefix}:{user_id}"
            
            # 尝试从缓存获取
            cached_result = api_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 调用原函数
            result = f(*args, **kwargs)
            
            # 缓存结果（仅缓存成功的响应）
            if isinstance(result, tuple):
                data, status_code = result if len(result) == 2 else (result, 200)
                if status_code == 200:
                    api_cache.set(cache_key, result)
            else:
                api_cache.set(cache_key, result)
            
            return result
        return decorated_function
    return decorator


def create_route_animation(td, actions, save_path, title="路线生成过程", fps=2):
    """
    创建TSP路线逐步生成的动态GIF
    
    参数:
        td: TensorDict，包含城市坐标等信息
        actions: numpy数组，访问城市的顺序
        save_path: GIF保存路径
        title: 图表标题
        fps: 帧率（每秒帧数）
    """
    # 提取城市坐标
    if hasattr(td, 'get'):
        locs = td.get('locs', td['locs']).cpu().numpy()
    else:
        locs = td['locs'].cpu().numpy()
    
    num_cities = len(locs)
    frames = []
    
    # 计算每一步的累计距离
    def calculate_partial_distance(locs, actions, step):
        """计算到第step步为止的累计距离"""
        if step < 1:
            return 0.0
        total_dist = 0.0
        for i in range(step):
            city_a = locs[actions[i]]
            # 如果是最后一步，返回起点；否则继续下一个城市
            if i + 1 < len(actions):
                city_b = locs[actions[i + 1]]
            else:
                city_b = locs[actions[0]]  # 返回起点
            dist = np.sqrt(np.sum((city_a - city_b) ** 2))
            total_dist += dist
        return total_dist
    
    # 为每一步生成一帧图像
    for step in range(num_cities + 1):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制所有城市点
        ax.scatter(locs[:, 0], locs[:, 1], c='lightblue', s=200, 
                  zorder=3, alpha=0.6, edgecolors='black', linewidths=2)
        
        # 标注城市编号
        for i, (x, y) in enumerate(locs):
            ax.text(x, y, str(i), fontsize=10, ha='center', va='center',
                   fontweight='bold', color='darkblue')
        
        # 绘制已经构建的路径
        if step > 0:
            for i in range(step):
                start = locs[actions[i]]
                if i + 1 < len(actions):
                    end = locs[actions[i + 1]]
                else:
                    end = locs[actions[0]]  # 最后返回起点
                
                # 绘制路径线
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       'b-', linewidth=3, alpha=0.7, zorder=1)
                
                # 添加箭头表示方向
                mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
                dx, dy = end[0] - start[0], end[1] - start[1]
                ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1), 
                          xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                          arrowprops=dict(arrowstyle='->', color='blue', 
                                        lw=2, alpha=0.7))
        
        # 高亮当前访问的城市
        if step > 0 and step <= num_cities:
            current_city = actions[step - 1]
            ax.scatter(locs[current_city, 0], locs[current_city, 1], 
                      c='red', s=400, zorder=5, marker='*', 
                      edgecolors='darkred', linewidths=2,
                      label=f'当前: 城市 {current_city}')
        
        # 高亮起点
        start_city = actions[0]
        ax.scatter(locs[start_city, 0], locs[start_city, 1], 
                  c='green', s=300, zorder=4, marker='s',
                  edgecolors='darkgreen', linewidths=2,
                  label=f'起点: 城市 {start_city}')
        
        # 计算当前累计成本
        current_cost = calculate_partial_distance(locs, actions, step)
        
        # 设置标题和信息
        if step == 0:
            info_text = "开始构建路线..."
        elif step < num_cities:
            info_text = f"第 {step} 步 | 已访问 {step} 个城市 | 累计成本: {current_cost:.3f}"
        else:
            # 最后一步，返回起点
            final_dist = np.sqrt(np.sum((locs[actions[-1]] - locs[actions[0]]) ** 2))
            total_cost = current_cost + final_dist
            info_text = f"完成！总共 {num_cities} 个城市 | 总成本: {total_cost:.3f}"
        
        ax.set_title(f"{title}\n{info_text}", fontsize=14, fontweight='bold', pad=20)
        
        # 设置坐标轴
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('X 坐标', fontsize=12)
        ax.set_ylabel('Y 坐标', fontsize=12)
        
        # 添加图例
        if step > 0:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # 添加进度条
        progress = step / num_cities
        ax.text(0.5, -0.12, f"进度: {int(progress * 100)}%", 
               ha='center', va='top', transform=ax.transAxes,
               fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 保存当前帧为图像
        fig.tight_layout()
        
        # 将图形转换为PIL Image（兼容新旧版本matplotlib）
        fig.canvas.draw()
        try:
            # 新版本 matplotlib (>= 3.8)
            buf = fig.canvas.buffer_rgba()
            image = np.frombuffer(buf, dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # 转换 RGBA 到 RGB
            image = image[:, :, :3]
        except AttributeError:
            # 旧版本 matplotlib
            try:
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # 更老的版本，使用 tostring_argb
                buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                # ARGB 转 RGB
                image = buf[:, :, 1:]
        
        frames.append(Image.fromarray(image))
        
        plt.close(fig)
    
    # 在最后一帧停留更长时间
    for _ in range(3):
        frames.append(frames[-1])
    
    # 保存为GIF
    duration = int(1000 / fps)  # 每帧持续时间（毫秒）
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False
    )

# ============================================
# 用户认证路由（新增）
# ============================================

@app.route('/')
def index():
    """主页 - 强制登录检查"""
    # 清理可能的无效session
    user_id = get_current_user_id()
    
    # 强制检查用户是否真实存在
    if not user_id:
        # 清除可能损坏的session
        session.clear()
        return redirect(url_for('login_page'))
    
    # 验证用户是否在数据库中存在
    user_manager = get_user_manager()
    if user_manager:
        user_info = user_manager.get_user(user_id)
        if not user_info:
            # 用户不存在，清除session并重定向
            session.clear()
            return redirect(url_for('login_page'))
    
    username = get_current_username()
    return render_template('index.html', 
                         is_logged_in=True, 
                         username=username,
                         active_page='home')

@app.route('/login')
def login_page():
    """登录页面 - 如果已登录则跳转到主页"""
    user_id = get_current_user_id()
    if user_id:
        # 已登录，直接跳转到主页
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/register')
def register_page():
    """注册页面"""
    return render_template('register.html')

@app.route('/api/register', methods=['POST'])
def api_register():
    """用户注册API"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        
        if not username or not password:
            return jsonify({
                'success': False, 
                'message': '用户名和密码不能为空'
            }), 400
        
        user_manager = get_user_manager()
        if user_manager is None:
            return jsonify({
                'success': False,
                'message': '认证模块未初始化'
            }), 500
        
        success, message, user_id = user_manager.create_user(username, password, email)
        
        if success:
            return jsonify({
                'success': True, 
                'message': message,
                'user_id': user_id
            })
        else:
            return jsonify({
                'success': False, 
                'message': message
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'注册失败：{str(e)}'
        }), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    """用户登录API"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({
                'success': False, 
                'message': '用户名和密码不能为空'
            }), 400
        
        user_manager = get_user_manager()
        if user_manager is None:
            return jsonify({
                'success': False,
                'message': '认证模块未初始化'
            }), 500
        
        success, message, user_data = user_manager.verify_user(username, password)
        
        if success:
            set_user_session(user_data)
            return jsonify({
                'success': True, 
                'message': message,
                'user': user_data
            })
        else:
            return jsonify({
                'success': False, 
                'message': message
            }), 401
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'登录失败：{str(e)}'
        }), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """用户登出API"""
    clear_user_session()
    return jsonify({
        'success': True, 
        'message': '已退出登录'
    })

@app.route('/logout')
def logout_page():
    """直接访问的登出页面 - 清除session并跳转到登录页"""
    session.clear()
    clear_user_session()
    return redirect(url_for('login_page'))

@app.route('/api/current_user', methods=['GET'])
def api_current_user():
    """获取当前登录用户信息"""
    user_id = get_current_user_id()
    if not user_id:
        return jsonify({
            'success': False, 
            'message': '未登录'
        }), 401
    
    user_manager = get_user_manager()
    if user_manager is None:
        return jsonify({
            'success': False,
            'message': '认证模块未初始化'
        }), 500
    
    user_data = user_manager.get_user(user_id)
    return jsonify({
        'success': True, 
        'user': user_data
    })

# ============================================
# 原有路由（保持兼容）
# ============================================

@app.route('/res')
def Index_res():
    """旧版注册页面 - 重定向到新页面"""
    return redirect(url_for('register_page'))

@app.route('/benchmark')
@login_required
def benchmark():
    """算法性能对比页面 - 需要登录"""
    return render_template('benchmark.html', active_page='benchmark')

@app.route('/file_manager')
@login_required
def file_manager():
    """文件管理页面 - 需要登录"""
    return render_template('file_manager.html', active_page='file_manager')

@app.route('/profile')
@login_required
def profile():
    """我的账户页面 - 需要登录"""
    user_id = get_current_user_id()
    username = get_current_username()
    
    # 获取用户详细信息
    user_data = {
        'username': username,
        'email': None,
        'create_time': None,
        'last_login': None
    }
    
    user_manager = get_user_manager()
    if user_manager:
        user_info = user_manager.get_user(user_id)
        if user_info:
            user_data['email'] = user_info.get('email')
            user_data['create_time'] = user_info.get('create_time').strftime('%Y-%m-%d %H:%M') if user_info.get('create_time') else None
            user_data['last_login'] = user_info.get('last_login').strftime('%Y-%m-%d %H:%M') if user_info.get('last_login') else None
    
    return render_template('profile.html', **user_data, active_page='profile')

@app.route('/api/user_stats', methods=['GET'])
@login_required
@cached_api(key_prefix='user_stats')
def get_user_stats():
    """获取用户统计数据 - 实时数据"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'success': False, 'message': '未登录'}), 401
        
        stats = {
            'total_sessions': 0,
            'completed_sessions': 0,
            'running_sessions': 0,
            'failed_sessions': 0,
            'total_files': 0,
            'total_size_mb': 0
        }
        
        # 从数据库获取统计数据
        db = get_db()
        session_manager = get_session_manager()
        file_manager = get_file_manager()
        
        if db and session_manager and file_manager:
            try:
                # 获取训练会话统计
                cursor = db.cursor(dictionary=True)
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                        SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) as running,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
                    FROM training_sessions
                    WHERE user_id = %s
                """, (user_id,))
                session_stats = cursor.fetchone()
                
                if session_stats:
                    stats['total_sessions'] = session_stats['total'] or 0
                    stats['completed_sessions'] = session_stats['completed'] or 0
                    stats['running_sessions'] = session_stats['running'] or 0
                    stats['failed_sessions'] = session_stats['failed'] or 0
                
                # 获取文件统计
                storage_stats = file_manager.get_user_storage_stats(user_id)
                if storage_stats:
                    stats['total_files'] = storage_stats['total_files'] or 0
                    stats['total_size_mb'] = storage_stats['total_mb'] or 0
                    
            except Exception as e:
                print(f"获取统计数据失败: {str(e)}")
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取统计失败: {str(e)}'
        }), 500

@app.route('/api/user_activity', methods=['GET'])
@login_required
@cached_api(key_prefix='user_activity')
def get_user_activity():
    """获取用户最近活动记录 - 实时数据"""
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'success': False, 'message': '未登录'}), 401
        
        activities = []
        
        # 从数据库获取最近的训练会话
        db = get_db()
        session_manager = get_session_manager()
        
        if db and session_manager:
            try:
                cursor = db.cursor(dictionary=True)
                cursor.execute("""
                    SELECT 
                        session_id,
                        model_type,
                        problem_type,
                        status,
                        start_time,
                        end_time,
                        final_reward
                    FROM training_sessions
                    WHERE user_id = %s
                    ORDER BY start_time DESC
                    LIMIT 10
                """, (user_id,))
                
                sessions = cursor.fetchall()
                
                for session in sessions:
                    time_str = session['start_time'].strftime('%Y-%m-%d %H:%M')
                    
                    if session['status'] == 'completed':
                        text = f"完成了 {session['problem_type'].upper()} 问题的训练 ({session['model_type']})"
                        if session['final_reward']:
                            text += f" - 奖励: {session['final_reward']:.2f}"
                    elif session['status'] == 'running':
                        text = f"开始训练 {session['model_type']} 模型 ({session['problem_type'].upper()})"
                    elif session['status'] == 'failed':
                        text = f"训练失败: {session['model_type']} ({session['problem_type'].upper()})"
                    else:
                        text = f"训练 {session['model_type']} - {session['status']}"
                    
                    activities.append({
                        'time': time_str,
                        'text': text,
                        'status': session['status']
                    })
                    
            except Exception as e:
                print(f"获取活动记录失败: {str(e)}")
        
        # 如果没有活动记录，返回提示
        if not activities:
            activities = [{
                'time': '无记录',
                'text': '还没有训练记录，开始你的第一次训练吧！',
                'status': 'info'
            }]
        
        return jsonify({
            'success': True,
            'activities': activities
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取活动记录失败: {str(e)}'
        }), 500

# 模型知识库数据
MODEL_DATABASE = {
    "AM": {
        "name": "AM",
        "full_name": "Attention Model - 注意力模型",
        "category": "构造方法（自回归）",
        "year": "2019",
        "paper_url": "https://arxiv.org/abs/1803.08475",
        "paper_venue": "ICLR 2019",
        "core_concept": """
            <p>Attention Model (AM) 是首个成功将Transformer架构应用于组合优化问题的深度强化学习模型。它利用注意力机制捕捉节点间的全局依赖关系，通过编码器-解码器结构逐步构建解决方案。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：将TSP等路由问题视为序列到序列(seq2seq)问题，使用多头注意力机制学习节点的重要性，并通过强化学习优化策略网络。
            </div>
        """,
        "architecture": """
            <h4>1. 编码器（Encoder）</h4>
            <p>使用Transformer编码器处理输入节点特征，生成节点的嵌入表示：</p>
            <ul>
                <li><strong>输入</strong>：节点坐标 (x, y) 或其他特征</li>
                <li><strong>多头注意力</strong>：捕捉节点间的全局关系</li>
                <li><strong>前馈网络</strong>：提取高层特征</li>
                <li><strong>输出</strong>：节点嵌入向量</li>
            </ul>
            
            <h4>2. 解码器（Decoder）</h4>
            <p>自回归地生成访问序列：</p>
            <ul>
                <li><strong>上下文嵌入</strong>：聚合已访问节点和当前状态</li>
                <li><strong>注意力评分</strong>：计算每个候选节点的选择概率</li>
                <li><strong>动作掩码</strong>：确保不重复访问已选节点</li>
                <li><strong>采样/贪心</strong>：根据概率分布选择下一个节点</li>
            </ul>
            
            <h4>3. 训练策略</h4>
            <p>使用REINFORCE算法进行策略梯度优化：</p>
            <ul>
                <li><strong>Baseline</strong>：使用贪心rollout或指数移动平均降低方差</li>
                <li><strong>Reward</strong>：路径总长度的负值</li>
                <li><strong>梯度估计</strong>：通过采样多条路径估计策略梯度</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>Transformer用于CO</strong>：首次将Transformer成功应用于组合优化</li>
                <li>🔹 <strong>无需监督数据</strong>：纯强化学习训练，不需要最优解标签</li>
                <li>🔹 <strong>问题无关架构</strong>：可轻松适配TSP、VRP等多种问题</li>
                <li>🔹 <strong>并行计算友好</strong>：Transformer结构支持高效GPU并行</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>TSP-50</h5>
                    <p>Gap: 1.41%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>TSP-100</h5>
                    <p>Gap: 1.73%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>CVRP-50</h5>
                    <p>Gap: 5.30%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>CVRP-100</h5>
                    <p>Gap: 3.39%<br>Time: <1s</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">在单次前向传播下，AM在TSP-50上达到1.41%的Gap，速度极快但质量略逊于后续改进方法。</p>
        """,
        "advantages": """
            <ul>
                <li>推理速度快（<1秒）</li>
                <li>架构简洁，易于理解和实现</li>
                <li>可扩展性好，支持不同规模问题</li>
                <li>泛化能力强，训练规模可迁移</li>
                <li>开创性工作，影响力大</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>单次解码质量有限（Gap~1-2%）</li>
                <li>未充分利用问题对称性</li>
                <li>训练需要大量样本</li>
                <li>对超参数较敏感</li>
            </ul>
        """,
        "applications": """
            <p>适用于需要快速求解的场景：</p>
            <ul>
                <li>实时物流规划</li>
                <li>在线路径优化</li>
                <li>大规模问题的快速近似</li>
                <li>作为其他方法的baseline</li>
            </ul>
        """
    },
    "POMO": {
        "name": "POMO",
        "full_name": "Policy Optimization with Multiple Optima",
        "category": "构造方法（自回归）",
        "year": "2021",
        "paper_url": "https://arxiv.org/abs/2010.16011",
        "paper_venue": "NeurIPS 2021",
        "core_concept": """
            <p>POMO 通过从不同起点同时构建多条路径来利用TSP等问题的对称性，显著提升了求解质量而不增加模型复杂度。</p>
            <div class="highlight-box">
                <strong>核心洞察</strong>：TSP问题具有旋转对称性 - 从任意节点出发都能得到等价的最优解。POMO利用这一特性，在训练和推理时同时考虑所有可能的起点。
            </div>
        """,
        "architecture": """
            <h4>1. 多起点并行解码</h4>
            <p>与AM的关键区别：</p>
            <ul>
                <li><strong>AM</strong>：固定从节点0开始</li>
                <li><strong>POMO</strong>：同时从所有N个节点开始，生成N条路径</li>
            </ul>
            
            <h4>2. 增强训练策略</h4>
            <p>训练时的优势：</p>
            <ul>
                <li>每个batch实际产生 N×batch_size 条路径</li>
                <li>所有路径共享梯度，加速学习</li>
                <li>利用对称性，减少训练方差</li>
            </ul>
            
            <h4>3. 推理时的策略</h4>
            <ul>
                <li><strong>训练模式</strong>：使用所有N个起点的平均损失</li>
                <li><strong>推理模式</strong>：取N条路径中的最优解</li>
                <li><strong>增强版本</strong>：可结合数据增强进一步提升（8×N条路径）</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>对称性利用</strong>：充分利用TSP的旋转不变性</li>
                <li>🔹 <strong>训练加速</strong>：N倍数据增强无额外计算成本</li>
                <li>🔹 <strong>推理提升</strong>：N条路径选最优，质量显著提高</li>
                <li>🔹 <strong>无架构修改</strong>：基于AM架构，无需重新设计</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>TSP-50 (Greedy)</h5>
                    <p>Gap: 0.89%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>TSP-50 (Sampling)</h5>
                    <p>Gap: 0.18%<br>Time: 1m</p>
                </div>
                <div class="info-card">
                    <h5>TSP-100 (Greedy)</h5>
                    <p>Gap: 0.05%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>CVRP-50</h5>
                    <p>Gap: 3.99%<br>Time: <1s</p>
                </div>
            </div>
            <p style="margin-top: 1rem;"><strong>显著优于AM</strong>：在TSP-50上从1.41%降至0.89%，接近50%的质量提升！</p>
        """,
        "advantages": """
            <ul>
                <li>质量大幅提升（AM的1.5-2倍）</li>
                <li>训练速度快（数据利用率高）</li>
                <li>推理仍然很快（<1秒）</li>
                <li>实现简单，基于AM小改</li>
                <li>适用于所有对称性问题</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>仅适用于具有对称性的问题</li>
                <li>GPU内存占用增加N倍</li>
                <li>对非对称问题无优势</li>
                <li>极大规模问题内存压力大</li>
            </ul>
        """,
        "applications": """
            <p>POMO特别适合：</p>
            <ul>
                <li>TSP及其变体（对称性强）</li>
                <li>CVRP等车辆路径问题</li>
                <li>需要高质量解的实时应用</li>
                <li>GPU资源充足的场景</li>
            </ul>
        """
    },
    "SymNCO": {
        "name": "Sym-NCO",
        "full_name": "Symmetric Neural Combinatorial Optimization",
        "category": "构造方法（自回归）",
        "year": "2022",
        "paper_url": "https://arxiv.org/abs/2205.13209",
        "paper_venue": "NeurIPS 2022",
        "core_concept": """
            <p>Sym-NCO 将对称性的利用从推理扩展到整个网络架构，通过设计等变神经网络来强制模型学习问题的内在对称结构。</p>
            <div class="highlight-box">
                <strong>核心创新</strong>：不仅像POMO那样在数据层面利用对称性，而是在网络层面嵌入对称性约束，使模型从根本上学习到旋转、翻转等对称不变的特征。
            </div>
        """,
        "architecture": """
            <h4>1. 等变网络设计</h4>
            <p>关键架构特点：</p>
            <ul>
                <li><strong>等变编码器</strong>：对输入的旋转/翻转变换，输出也相应变换</li>
                <li><strong>不变特征</strong>：提取对称不变的全局特征</li>
                <li><strong>条件解码</strong>：基于不变特征的条件生成</li>
            </ul>
            
            <h4>2. 对称性分类</h4>
            <p>Sym-NCO区分并利用三类对称性：</p>
            <ul>
                <li><strong>旋转对称</strong>：任意节点可作为起点</li>
                <li><strong>翻转对称</strong>：顺时针/逆时针路径等价</li>
                <li><strong>排列对称</strong>：节点标签可任意排列</li>
            </ul>
            
            <h4>3. 训练优化</h4>
            <ul>
                <li>对称增强的数据生成</li>
                <li>等变性损失约束</li>
                <li>多起点联合训练</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>等变网络架构</strong>：从网络层面保证对称性</li>
                <li>🔹 <strong>理论保证</strong>：严格的数学对称性约束</li>
                <li>🔹 <strong>泛化能力</strong>：更好的分布外泛化</li>
                <li>🔹 <strong>参数效率</strong>：通过对称性减少需要学习的参数</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>TSP-50 (Greedy)</h5>
                    <p>Gap: 0.47%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>TSP-50 (Aug)</h5>
                    <p>Gap: 0.01%<br>Time: 1m</p>
                </div>
                <div class="info-card">
                    <h5>TSP-20</h5>
                    <p>Gap: 0.05%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>CVRP-50</h5>
                    <p>Gap: 4.61%<br>Time: <1s</p>
                </div>
            </div>
            <p style="margin-top: 1rem;"><strong>目前最优</strong>：在TSP-50上达到0.47% Gap（Greedy），使用数据增强后仅0.01%，几乎最优！</p>
        """,
        "advantages": """
            <ul>
                <li>质量最优（当前SOTA之一）</li>
                <li>理论基础扎实</li>
                <li>泛化能力强</li>
                <li>参数效率高</li>
                <li>训练稳定性好</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>实现复杂度高</li>
                <li>需要深入理解群论</li>
                <li>计算开销略高于AM/POMO</li>
                <li>对非对称问题适用性有限</li>
            </ul>
        """,
        "applications": """
            <p>Sym-NCO最适合：</p>
            <ul>
                <li>对解质量要求极高的场景</li>
                <li>需要分布外泛化的应用</li>
                <li>学术研究和方法对比</li>
                <li>对称性强的CO问题</li>
            </ul>
        """
    },
    "REINFORCE": {
        "name": "REINFORCE",
        "full_name": "REINFORCE Algorithm",
        "category": "强化学习算法",
        "year": "1992",
        "paper_url": "https://link.springer.com/article/10.1007/BF00992696",
        "paper_venue": "Machine Learning 1992",
        "core_concept": """
            <p>REINFORCE 是最经典的策略梯度算法，直接优化策略网络的参数以最大化期望回报。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：通过蒙特卡洛采样估计策略梯度，根据实际获得的回报调整策略，使好的动作更可能被选择。
            </div>
        """,
        "architecture": """
            <h4>算法流程</h4>
            <ol>
                <li><strong>采样</strong>：根据当前策略π生成完整的轨迹</li>
                <li><strong>计算回报</strong>：R = -路径长度（TSP情况）</li>
                <li><strong>计算梯度</strong>：∇J = E[∇log π(a|s) · (R - b)]</li>
                <li><strong>更新参数</strong>：θ ← θ + α∇J</li>
            </ol>
            
            <h4>Baseline技巧</h4>
            <p>为降低梯度方差，使用baseline b：</p>
            <ul>
                <li><strong>移动平均</strong>：b = EMA(R)</li>
                <li><strong>Critic网络</strong>：b = V(s)</li>
                <li><strong>Greedy rollout</strong>：b = 贪心解的回报</li>
            </ul>
        """,
        "performance": """
            <p>REINFORCE本身是训练算法，不是模型架构。它被用于训练AM、POMO等模型。</p>
            <p>配合AM使用时的性能参考AM的数据。</p>
        """,
        "advantages": """
            <ul>
                <li>简单直观，易于实现</li>
                <li>适用于任意策略网络</li>
                <li>无需值函数近似</li>
                <li>适合高维离散动作空间</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>梯度方差大</li>
                <li>样本效率低</li>
                <li>训练不稳定</li>
                <li>需要大量episode</li>
            </ul>
        """
    },
    "DeepACO": {
        "name": "DeepACO",
        "full_name": "Deep Ant Colony Optimization",
        "category": "构造方法（非自回归）",
        "year": "2023",
        "paper_url": "https://arxiv.org/abs/2309.14032",
        "paper_venue": "NeurIPS 2023",
        "core_concept": """
            <p>DeepACO 将经典的蚁群优化算法与深度学习相结合，使用神经网络学习启发式信息，指导蚁群的路径搜索。</p>
            <div class="highlight-box">
                <strong>核心创新</strong>：用神经网络替代传统ACO的启发式函数，使算法能够从数据中学习问题特定的搜索策略，兼具ACO的全局搜索能力和深度学习的表征能力。
            </div>
        """,
        "architecture": """
            <h4>1. 神经网络启发式</h4>
            <ul>
                <li>使用GNN学习边的启发式值</li>
                <li>替代传统的距离倒数启发式</li>
                <li>能够捕捉复杂的问题结构</li>
            </ul>
            
            <h4>2. 蚁群搜索</h4>
            <ul>
                <li>多只蚂蚁并行构建解</li>
                <li>信息素更新机制</li>
                <li>局部搜索优化</li>
            </ul>
            
            <h4>3. 非自回归特性</h4>
            <ul>
                <li>所有蚂蚁同时构建解</li>
                <li>并行度高，速度快</li>
                <li>适合大规模问题</li>
            </ul>
        """,
        "performance": """
            <p>DeepACO在TSP上表现优异，特别是在大规模实例上。</p>
            <div class="info-card">
                <h5>特点</h5>
                <p>• 质量高（接近最优）<br>• 大规模问题优势明显<br>• 可解释性强</p>
            </div>
        """,
        "advantages": """
            <ul>
                <li>解质量高</li>
                <li>可解释性强（基于ACO）</li>
                <li>大规模问题表现好</li>
                <li>结合深度学习和传统优化</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>推理时间较长</li>
                <li>需要多次迭代</li>
                <li>实现复杂</li>
                <li>超参数较多</li>
            </ul>
        """
    },
    "MatNet": {
        "name": "MatNet",
        "full_name": "Matrix Network - 矩阵网络",
        "category": "构造方法（自回归）",
        "year": "2021",
        "paper_url": "https://arxiv.org/abs/2106.11113",
        "paper_venue": "NeurIPS 2021",
        "core_concept": """
            <p>MatNet 通过直接建模节点对之间的关系矩阵，实现了更强的表达能力和更好的性能。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：传统方法对每个节点独立编码，而MatNet使用矩阵表示节点对之间的关系，能够更好地捕捉图结构信息。
            </div>
        """,
        "architecture": """
            <h4>1. 矩阵编码器</h4>
            <ul>
                <li>构建节点对关系矩阵</li>
                <li>使用矩阵注意力机制</li>
                <li>捕捉高阶结构信息</li>
            </ul>
            
            <h4>2. 解码策略</h4>
            <ul>
                <li>基于矩阵表示的动作选择</li>
                <li>考虑已选路径的全局信息</li>
                <li>动态更新关系矩阵</li>
            </ul>
        """,
        "performance": """
            <p>MatNet在TSP和VRP问题上都表现出色，特别是在大规模问题上优势明显。</p>
        """,
        "advantages": """
            <ul>
                <li>表达能力强</li>
                <li>性能优异</li>
                <li>适用于多种问题</li>
                <li>对图结构建模充分</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>计算复杂度高（O(n²)）</li>
                <li>内存占用大</li>
                <li>训练时间较长</li>
                <li>实现复杂</li>
            </ul>
        """
    },
    "A2C": {
        "name": "A2C",
        "full_name": "Advantage Actor-Critic",
        "category": "强化学习算法",
        "year": "2016",
        "paper_url": "https://arxiv.org/abs/1602.01783",
        "paper_venue": "ICML 2016",
        "core_concept": """
            <p>A2C 是Actor-Critic算法的同步版本，同时学习策略网络（Actor）和价值网络（Critic）。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：使用Critic网络估计状态价值，为Actor提供更稳定的训练信号，减少REINFORCE的方差问题。
            </div>
        """,
        "architecture": """
            <h4>算法组成</h4>
            <ul>
                <li><strong>Actor</strong>：策略网络π(a|s)</li>
                <li><strong>Critic</strong>：价值网络V(s)</li>
                <li><strong>Advantage</strong>：A(s,a) = R - V(s)</li>
            </ul>
            
            <h4>训练流程</h4>
            <ol>
                <li>Actor生成动作并执行</li>
                <li>Critic评估状态价值</li>
                <li>计算优势函数</li>
                <li>同时更新Actor和Critic</li>
            </ol>
        """,
        "performance": """
            <p>A2C在训练稳定性和样本效率上优于REINFORCE。</p>
        """,
        "advantages": """
            <ul>
                <li>训练更稳定</li>
                <li>方差更小</li>
                <li>收敛更快</li>
                <li>样本效率高</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>需要额外的Critic网络</li>
                <li>实现复杂度增加</li>
                <li>超参数更多</li>
                <li>可能陷入局部最优</li>
            </ul>
        """
    },
    "PPO": {
        "name": "PPO",
        "full_name": "Proximal Policy Optimization",
        "category": "强化学习算法",
        "year": "2017",
        "paper_url": "https://arxiv.org/abs/1707.06347",
        "paper_venue": "ArXiv 2017",
        "core_concept": """
            <p>PPO 通过限制策略更新幅度来平衡探索与开发，是目前最流行的策略梯度算法之一。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：使用裁剪目标函数，防止策略更新步长过大，确保训练稳定性。
            </div>
        """,
        "architecture": """
            <h4>PPO-Clip目标</h4>
            <p>L(θ) = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]</p>
            <ul>
                <li><strong>r(θ)</strong>：新旧策略的概率比</li>
                <li><strong>clip</strong>：限制在[1-ε, 1+ε]范围内</li>
                <li><strong>A</strong>：优势函数</li>
            </ul>
            
            <h4>训练特点</h4>
            <ul>
                <li>多次利用同一批数据</li>
                <li>自适应调整学习率</li>
                <li>稳定的策略改进</li>
            </ul>
        """,
        "performance": """
            <p>PPO在各种RL任务上都表现优异，被认为是最可靠的策略梯度算法。</p>
        """,
        "advantages": """
            <ul>
                <li>训练极其稳定</li>
                <li>样本效率高</li>
                <li>超参数不敏感</li>
                <li>实现相对简单</li>
                <li>工业界广泛使用</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>计算开销较大</li>
                <li>需要多次迭代</li>
                <li>墙钟时间较长</li>
            </ul>
        """
    },
    "PtrNet": {
        "name": "PtrNet",
        "full_name": "Pointer Network - 指针网络",
        "category": "构造方法（自回归）",
        "year": "2015",
        "paper_url": "https://arxiv.org/abs/1506.03134",
        "paper_venue": "NeurIPS 2015",
        "core_concept": """
            <p>Pointer Network 是最早将seq2seq模型应用于组合优化的开创性工作。</p>
            <div class="highlight-box">
                <strong>核心创新</strong>：输出层不是固定词表，而是"指向"输入序列中的元素，天然适合TSP等排列问题。
            </div>
        """,
        "architecture": """
            <h4>架构特点</h4>
            <ul>
                <li>LSTM编码器处理输入</li>
                <li>注意力机制指向输入节点</li>
                <li>自回归生成访问序列</li>
            </ul>
        """,
        "performance": """
            <p>作为早期工作，性能不如现代Transformer based方法，但具有重要的历史意义。</p>
        """,
        "advantages": """
            <ul>
                <li>开创性工作</li>
                <li>概念简洁清晰</li>
                <li>易于理解</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>LSTM序列化处理慢</li>
                <li>性能不如现代方法</li>
                <li>难以并行化</li>
            </ul>
        """
    },
    "HAM": {
        "name": "HAM",
        "full_name": "Hierarchical Attention Model",
        "category": "构造方法（自回归）",
        "year": "2020",
        "paper_url": "https://arxiv.org/abs/2011.03227",
        "paper_venue": "AAAI 2021",
        "core_concept": """
            <p>HAM 引入层次化注意力机制，在不同粒度上捕捉问题结构，通过全局和局部两个层次的注意力实现更精细的节点特征建模。</p>
            <div class="highlight-box">
                <strong>核心创新</strong>：传统方法使用单一尺度的注意力，HAM通过层次化设计同时捕捉全局结构信息和局部邻域关系，提升了模型的表达能力。
            </div>
        """,
        "architecture": """
            <h4>1. 全局注意力层</h4>
            <ul>
                <li><strong>作用范围</strong>：考虑所有节点的关系</li>
                <li><strong>功能</strong>：捕捉问题的整体结构和远距离依赖</li>
                <li><strong>实现</strong>：Multi-head Transformer Attention</li>
            </ul>
            
            <h4>2. 局部注意力层</h4>
            <ul>
                <li><strong>作用范围</strong>：关注空间上相近的节点</li>
                <li><strong>功能</strong>：提取局部区域的细粒度特征</li>
                <li><strong>实现</strong>：基于距离的受限注意力机制</li>
            </ul>
            
            <h4>3. 层次化融合</h4>
            <ul>
                <li>门控机制动态平衡全局和局部特征</li>
                <li>自适应权重分配</li>
                <li>残差连接保证梯度流动</li>
            </ul>
            
            <h4>4. 解码策略</h4>
            <ul>
                <li>基于融合特征的自回归解码</li>
                <li>动态上下文更新</li>
                <li>Masked attention防止重复访问</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>双层注意力</strong>：全局+局部的层次化设计</li>
                <li>🔹 <strong>自适应融合</strong>：门控机制动态平衡不同尺度</li>
                <li>🔹 <strong>计算效率</strong>：局部注意力降低复杂度</li>
                <li>🔹 <strong>泛化能力</strong>：多尺度特征增强鲁棒性</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>TSP-50</h5>
                    <p>Gap: 1.15%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>TSP-100</h5>
                    <p>Gap: 1.52%<br>Time: 1s</p>
                </div>
                <div class="info-card">
                    <h5>CVRP-100</h5>
                    <p>Gap: 4.89%<br>Time: 2s</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">HAM在单次解码质量上优于基础AM，但略逊于POMO和Sym-NCO。</p>
        """,
        "advantages": """
            <ul>
                <li>多尺度特征建模能力强</li>
                <li>性能优于基础AM</li>
                <li>对复杂结构问题表现好</li>
                <li>局部注意力降低了计算复杂度</li>
                <li>泛化能力较强</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>实现复杂度高于AM</li>
                <li>训练难度增加</li>
                <li>超参数调优较敏感</li>
                <li>内存占用略高</li>
            </ul>
        """,
        "applications": """
            <p>HAM适用于：</p>
            <ul>
                <li>具有明显层次结构的CO问题</li>
                <li>大规模问题实例（受益于局部注意力）</li>
                <li>需要平衡质量和速度的场景</li>
                <li>区域性特征明显的路径问题</li>
            </ul>
        """
    },
    "PolyNet": {
        "name": "PolyNet",
        "full_name": "Polynomial Time Network",
        "category": "构造方法（自回归）",
        "year": "2021",
        "paper_url": "https://arxiv.org/abs/2102.09544",
        "paper_venue": "ICML 2021",
        "core_concept": """
            <p>PolyNet 通过指数移动平均（EMA）和Polyak平均技术来稳定神经网络训练，在组合优化中实现更平滑的收敛过程。</p>
            <div class="highlight-box">
                <strong>核心思想</strong>：维护模型参数的移动平均副本，在推理时使用平均后的参数，显著提升模型稳定性和泛化能力。
            </div>
        """,
        "architecture": """
            <h4>1. Polyak平均机制</h4>
            <ul>
                <li><strong>参数平滑</strong>：θ̄ = βθ̄ + (1-β)θ</li>
                <li><strong>训练参数</strong>：使用标准梯度更新</li>
                <li><strong>推理参数</strong>：使用平均后的θ̄</li>
            </ul>
            
            <h4>2. 双模型架构</h4>
            <ul>
                <li><strong>Online模型</strong>：接收梯度更新</li>
                <li><strong>Target模型</strong>：平滑参数副本</li>
                <li><strong>定期同步</strong>：每N步更新一次target模型</li>
            </ul>
            
            <h4>3. 训练策略</h4>
            <ul>
                <li>使用online模型进行前向传播和梯度计算</li>
                <li>Target模型用于生成baseline</li>
                <li>减少训练过程中的振荡</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>参数平滑</strong>：EMA降低训练噪声</li>
                <li>🔹 <strong>稳定Baseline</strong>：平均模型提供更稳定的baseline</li>
                <li>🔹 <strong>泛化提升</strong>：平均参数具有更好的泛化性</li>
                <li>🔹 <strong>即插即用</strong>：可应用于任何RL算法</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>TSP-50</h5>
                    <p>Gap: 1.20%<br>稳定性: ↑↑</p>
                </div>
                <div class="info-card">
                    <h5>训练收敛</h5>
                    <p>速度提升: 20%<br>方差降低: 30%</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">PolyNet主要优势在于训练稳定性，而非最终解质量的提升。</p>
        """,
        "advantages": """
            <ul>
                <li>训练极其稳定</li>
                <li>收敛曲线平滑</li>
                <li>减少训练方差</li>
                <li>泛化能力强</li>
                <li>实现简单</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>额外内存开销（双模型）</li>
                <li>需要调整平均系数β</li>
                <li>对最终性能提升有限</li>
                <li>主要改善训练过程</li>
            </ul>
        """,
        "applications": """
            <p>PolyNet特别适合：</p>
            <ul>
                <li>训练不稳定的大模型</li>
                <li>需要稳定训练的生产环境</li>
                <li>作为其他方法的补充技术</li>
                <li>超参数敏感的场景</li>
            </ul>
        """
    },
    "MTPOMO": {
        "name": "MTPOMO",
        "full_name": "Multi-Task Policy Optimization with Multiple Optima",
        "category": "构造方法（自回归）",
        "year": "2022",
        "paper_url": "https://arxiv.org/abs/2204.03236",
        "paper_venue": "NeurIPS 2022",
        "core_concept": """
            <p>MTPOMO 将POMO扩展到多任务学习场景，通过共享编码器同时学习TSP、CVRP、OP等多种组合优化问题，实现跨任务知识迁移和训练效率提升。</p>
            <div class="highlight-box">
                <strong>核心创新</strong>：使用统一的编码器提取问题无关的特征，针对不同任务使用专用的解码头，在多任务间共享知识，提升训练效率和泛化能力。
            </div>
        """,
        "architecture": """
            <h4>1. 共享编码器</h4>
            <ul>
                <li><strong>Transformer编码器</strong>：处理所有任务的输入</li>
                <li><strong>任务无关特征</strong>：提取通用的节点和图结构特征</li>
                <li><strong>参数共享</strong>：所有任务共享编码器权重</li>
            </ul>
            
            <h4>2. 任务专用解码器</h4>
            <ul>
                <li><strong>TSP解码器</strong>：针对旅行商问题的策略网络</li>
                <li><strong>CVRP解码器</strong>：处理容量约束的路径问题</li>
                <li><strong>OP解码器</strong>：针对定向问题的解码逻辑</li>
            </ul>
            
            <h4>3. 多任务训练策略</h4>
            <ul>
                <li>任务采样：每个batch随机选择任务</li>
                <li>损失平衡：动态调整各任务的权重</li>
                <li>梯度归一化：防止某个任务主导训练</li>
            </ul>
        """,
        "innovations": """
            <ul>
                <li>🔹 <strong>跨任务学习</strong>：首次将POMO扩展到多任务场景</li>
                <li>🔹 <strong>知识迁移</strong>：任务间共享编码器知识</li>
                <li>🔹 <strong>训练效率</strong>：一个模型解决多个问题</li>
                <li>🔹 <strong>零样本泛化</strong>：训练过的模型可适应新任务</li>
            </ul>
        """,
        "performance": """
            <div class="info-grid">
                <div class="info-card">
                    <h5>TSP-50</h5>
                    <p>Gap: 0.92%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>CVRP-50</h5>
                    <p>Gap: 4.12%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>OP-50</h5>
                    <p>Gap: 2.31%<br>Time: <1s</p>
                </div>
                <div class="info-card">
                    <h5>训练效率</h5>
                    <p>相比单任务<br>提升: 3x</p>
                </div>
            </div>
            <p style="margin-top: 1rem;">MTPOMO在各任务上的性能接近专用模型，但训练效率提升显著。</p>
        """,
        "advantages": """
            <ul>
                <li>一个模型解决多个问题</li>
                <li>训练效率高（3倍提升）</li>
                <li>跨任务知识迁移</li>
                <li>泛化能力强</li>
                <li>部署成本低</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>任务平衡困难（某些任务可能被忽视）</li>
                <li>内存占用大（多个解码器）</li>
                <li>单任务性能略逊于专用模型</li>
                <li>任务数量增加时扩展性有限</li>
            </ul>
        """,
        "applications": """
            <p>MTPOMO特别适合：</p>
            <ul>
                <li>需要解决多种CO问题的生产环境</li>
                <li>计算资源有限但问题类型多样</li>
                <li>快速原型开发和测试</li>
                <li>研究跨任务知识迁移</li>
            </ul>
        """
    },
    "MVMoE": {
        "name": "MVMoE",
        "full_name": "Multi-View Mixture of Experts",
        "category": "构造方法（自回归）",
        "year": "2023",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>MVMoE 使用混合专家模型，针对不同类型的问题实例使用不同的专家网络。</p>
        """,
        "advantages": """
            <ul>
                <li>适应性强</li>
                <li>专业化处理</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>模型复杂</li>
                <li>训练困难</li>
            </ul>
        """
    },
    "L2D": {
        "name": "L2D",
        "full_name": "Learn to Delegate",
        "category": "构造方法（自回归）",
        "year": "2023",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>L2D 学习何时使用神经网络求解，何时委托给传统启发式算法。</p>
        """,
        "advantages": """
            <ul>
                <li>灵活性高</li>
                <li>结合优势</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>决策复杂</li>
            </ul>
        """
    },
    "HGNN": {
        "name": "HGNN",
        "full_name": "Heterogeneous Graph Neural Network",
        "category": "构造方法（自回归）",
        "year": "2022",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>HGNN 使用异构图神经网络建模不同类型节点和边的关系。</p>
        """,
        "advantages": """
            <ul>
                <li>表达力强</li>
                <li>适用复杂问题</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>计算开销大</li>
            </ul>
        """
    },
    "DF": {
        "name": "DF",
        "full_name": "Distribution Fitting",
        "category": "构造方法（自回归）",
        "year": "2023",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>DF 通过拟合最优解的分布来生成高质量解。</p>
        """,
        "advantages": """
            <ul>
                <li>理论优雅</li>
                <li>性能优秀</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>训练复杂</li>
            </ul>
        """
    },
    "GFACS": {
        "name": "GFACS",
        "full_name": "Graph-based Fast Adaptive Construction Solver",
        "category": "构造方法（非自回归）",
        "year": "2023",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>GFACS 使用图神经网络非自回归地构建解。</p>
        """,
        "advantages": """
            <ul>
                <li>速度极快</li>
                <li>完全并行</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>质量可能不如自回归方法</li>
            </ul>
        """
    },
    "GLOP": {
        "name": "GLOP",
        "full_name": "Generalized Learning for Optimization Problems",
        "category": "构造方法（非自回归）",
        "year": "2023",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>GLOP 是一个通用的学习框架，适用于多种优化问题。</p>
        """,
        "advantages": """
            <ul>
                <li>通用性强</li>
                <li>易于扩展</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>特定问题性能可能不如专用方法</li>
            </ul>
        """
    },
    "DACT": {
        "name": "DACT",
        "full_name": "Dual Attention with Cross Transformation",
        "category": "改进方法",
        "year": "2022",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>DACT 使用双重注意力机制和交叉变换来改进现有模型。</p>
        """,
        "advantages": """
            <ul>
                <li>即插即用</li>
                <li>性能提升明显</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>计算开销增加</li>
            </ul>
        """
    },
    "N2S": {
        "name": "N2S",
        "full_name": "Neural to Symbolic",
        "category": "改进方法",
        "year": "2023",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>N2S 将神经网络的输出转换为符号化的优化算法。</p>
        """,
        "advantages": """
            <ul>
                <li>可解释性强</li>
                <li>质量优秀</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>转换过程复杂</li>
            </ul>
        """
    },
    "NeuOpt": {
        "name": "NeuOpt",
        "full_name": "Neural Optimizer",
        "category": "改进方法",
        "year": "2023",
        "paper_url": "#",
        "paper_venue": "Research Paper",
        "core_concept": """
            <p>NeuOpt 使用神经网络学习优化算法本身。</p>
        """,
        "advantages": """
            <ul>
                <li>学习能力强</li>
                <li>适应性好</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>训练复杂</li>
                <li>泛化挑战</li>
            </ul>
        """
    },
    "ActiveSearch": {
        "name": "ActiveSearch",
        "full_name": "Active Search",
        "category": "传导式强化学习",
        "year": "2020",
        "paper_url": "https://arxiv.org/abs/2012.05417",
        "paper_venue": "ICLR 2021",
        "core_concept": """
            <p>Active Search 在测试时继续优化策略，通过主动搜索改进解质量。</p>
        """,
        "advantages": """
            <ul>
                <li>测试时优化</li>
                <li>质量提升显著</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>推理时间长</li>
                <li>计算资源需求高</li>
            </ul>
        """
    },
    "EAS": {
        "name": "EAS",
        "full_name": "Efficient Active Search",
        "category": "传导式强化学习",
        "year": "2021",
        "paper_url": "https://arxiv.org/abs/2106.05126",
        "paper_venue": "NeurIPS 2021",
        "core_concept": """
            <p>EAS 是Active Search的高效版本，减少了测试时优化的计算开销。</p>
        """,
        "advantages": """
            <ul>
                <li>更快的测试时优化</li>
                <li>效率与质量平衡好</li>
            </ul>
        """,
        "limitations": """
            <ul>
                <li>仍需额外计算</li>
            </ul>
        """
    }
}

@app.route('/model_info')
@login_required
def model_info_list():
    """模型知识库列表页面"""
    # 获取所有分类
    categories = set()
    for model in MODEL_DATABASE.values():
        if 'category' in model:
            categories.add(model['category'])
    
    return render_template('model_list.html', 
                         models=MODEL_DATABASE, 
                         categories=sorted(categories),
                         active_page='model_info')

@app.route('/model_info/<model_id>')
@login_required
def model_info_detail(model_id):
    """模型详情页面"""
    if model_id not in MODEL_DATABASE:
        return "模型不存在", 404
    
    model_data = MODEL_DATABASE[model_id]
    return render_template('model_info.html', model_data=model_data, active_page='model_info')

# ============================================
# 旧的 register 和 login 路由已被新的 API 替代，已删除 ==========
#   新路由在文件开头：/api/register, /api/login, /api/logout


# 自定义 Lightning Callback 用于捕获训练进度
class ProgressCallback(Callback):  # 定义一个回调用于在训练过程中收集与推送指标
    def __init__(self, queue, session_id, total_epochs, user_id):  # ========== 添加user_id参数 ==========
        super().__init__()  # 调用父类初始化
        self.queue = queue  # 保存与前端通信的消息队列
        self.session_id = session_id  # 保存当前训练会话ID
        self.total_epochs = total_epochs  # 保存总训练轮数，用于百分比计算
        self.user_id = user_id  # ========== 保存用户ID ==========
        self.best_reward = float('-inf')  # 记录历史最优奖励（越大越好）
        self.epoch_losses = []  # 存放当前epoch内每个batch的loss
        self.epoch_rewards = []  # 存放当前epoch内每个batch的reward
        # 新增：用于存储所有epoch的历史数据，用于绘制折线图
        self.history_losses = []  # 所有epoch的平均loss历史
        self.history_rewards = []  # 所有epoch的平均reward历史
        self.history_epochs = []  # epoch编号列表
        # 为后台线程创建独立的数据库连接和管理器
        self.db = None
        self.file_manager = None
    
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
        
        # 新增：记录历史数据用于绘制折线图
        self.history_epochs.append(epoch)
        self.history_losses.append(loss)
        self.history_rewards.append(reward)
        
        # 新增：生成实时训练曲线图
        try:
            USER_PLOTS_DIR = get_user_plot_dir(self.user_id)  # 获取用户专属目录
            plot_filename = f"training_curves_{self.session_id[:8]}.png"
            plot_path = os.path.join(USER_PLOTS_DIR, plot_filename)  # ========== 使用用户目录 ==========
            
            # 创建包含loss和reward的双子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # 绘制Loss曲线
            ax1.plot(self.history_epochs, self.history_losses, 'b-o', linewidth=2, markersize=6, label='Loss')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('训练Loss变化曲线', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(loc='upper right', fontsize=10)
            
            # 绘制Reward曲线
            ax2.plot(self.history_epochs, self.history_rewards, 'g-o', linewidth=2, markersize=6, label='Reward')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Reward', fontsize=12)
            ax2.set_title('训练Reward变化曲线', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(loc='lower right', fontsize=10)
            
            # 在reward图上标注最佳reward
            best_epoch_idx = self.history_rewards.index(max(self.history_rewards))
            best_epoch_num = self.history_epochs[best_epoch_idx]
            ax2.axhline(y=self.best_reward, color='r', linestyle='--', alpha=0.5, label=f'Best: {self.best_reward:.4f}')
            ax2.scatter([best_epoch_num], [self.best_reward], color='red', s=100, zorder=5, marker='*')
            ax2.legend(loc='lower right', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            # ========== 保存文件记录到数据库 ==========
            if self.file_manager is None:
                self.db = get_background_db()
                if self.db:
                    self.file_manager = FileManager(self.db)
            
            if self.file_manager:
                try:
                    self.file_manager.save_file_record(
                        user_id=self.user_id,
                        session_id=self.session_id,
                        filename=plot_filename,
                        file_type='curve',
                        file_path=plot_path
                    )
                except Exception as e:
                    print(f"保存文件记录失败: {str(e)}")
            
            # 通过队列发送图表路径
            self.queue.put(json.dumps({
                'type': 'plot',
                'plot_url': f"/static/model_plots/user_{self.user_id}/{plot_filename}",  # ========== 使用用户目录 ==========
                'message': f'Epoch {epoch} 训练曲线已更新'
            }))
        except Exception as e:
            self.queue.put(json.dumps({
                'type': 'warning',
                'message': f'生成训练曲线失败: {str(e)}'
            }))
        
        # 更新训练状态  # 将最新指标写入全局状态，供查询接口使用
        training_status[self.session_id].update({
            'progress': progress,  # 当前进度百分比
            'epoch': epoch,  # 当前epoch编号
            'loss': round(loss, 4),  # 本epoch平均loss（四舍五入）
            'reward': round(reward, 4),  # 本epoch平均reward（四舍五入）
            'best_reward': round(self.best_reward, 4),  # 历史最优reward（四舍五入）
            'plot_url': f"/static/model_plots/user_{self.user_id}/training_curves_{self.session_id[:8]}.png"  # ========== 使用用户目录 ==========
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
def real_rl4co_training(config, session_id, user_id):  # ========== 添加user_id参数 ==========
    """使用 RL4CO 进行真实的强化学习训练"""  # 函数说明：真实训练模式
    queue = training_queues[session_id]  # 取出当前会话的消息队列
    
    # ========== 为后台线程创建独立的数据库连接 ==========
    bg_db = get_background_db()
    bg_session_manager = TrainingSessionManager(bg_db) if bg_db else None
    bg_file_manager = FileManager(bg_db) if bg_db else None
    
    try:  # 捕获训练过程中的异常
        # ========== 创建用户专属目录 ==========
        USER_PLOTS_DIR = get_user_plot_dir(user_id)
        USER_CHECKPOINTS_DIR = get_user_checkpoint_dir(user_id)
        
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
        checkpoint_path = os.path.join(USER_CHECKPOINTS_DIR, f"{problem_type}-{model_type}.ckpt")  # ========== 使用用户目录 ==========
        ckpt_path = checkpoint_path if os.path.exists(checkpoint_path) else None  # 若存在则使用
        
        if ckpt_path:  # 如果找到了历史ckpt
            queue.put(json.dumps({
                'type': 'info',  # 消息类型
                'message': f'加载检查点: {checkpoint_path}'  # 提示已加载
            }))
        
        # 创建进度回调  # 构建自定义回调以推送指标
        progress_callback = ProgressCallback(queue, session_id, epochs, user_id)  # ========== 传入user_id ==========
        
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
        animation_paths = []  # 存储动态GIF的路径
        
        for i, td in enumerate(td_init):  # 遍历每个测试实例
            # 生成静态对比图
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 创建左右两个子图
            env.render(td, actions_untrained[i], ax=axs[0])  # 左图渲染随机策略路径
            env.render(td, actions_trained[i], ax=axs[1])  # 右图渲染训练后策略路径
            axs[0].set_title(f"Random | Cost = {-rewards_untrained[i].item():.3f}")  # 左图标题：随机策略成本
            axs[1].set_title(f"Trained | Cost = {-rewards_trained[i].item():.3f}")  # 右图标题：训练后成本
            
            plot_filename = f"comparison_{session_id[:8]}_{i+1}.png"  # 生成图片文件名（含会话前缀）
            plot_path = os.path.join(USER_PLOTS_DIR, plot_filename)  # ========== 使用用户目录 ==========
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")  # 保存图片到磁盘
            plt.close()  # 关闭图像以释放内存
            
            # ========== 保存文件记录到数据库 ==========
            if bg_file_manager:
                try:
                    bg_file_manager.save_file_record(
                        user_id=user_id,
                        session_id=session_id,
                        filename=plot_filename,
                        file_type='plot',
                        file_path=plot_path
                    )
                except Exception as e:
                    print(f"保存文件记录失败: {str(e)}")
            
            plot_paths.append(f"/static/model_plots/user_{user_id}/{plot_filename}")  # ========== 使用用户目录 ==========
            
            # 生成动态路线构建过程GIF
            queue.put(json.dumps({
                'type': 'info',
                'message': f'正在生成动态路线图 {i+1}/3...'
            }))
            
            animation_filename = f"animation_{session_id[:8]}_{i+1}.gif"
            animation_path = os.path.join(USER_PLOTS_DIR, animation_filename)  # ========== 使用用户目录 ==========
            
            # 生成训练后路线的逐步构建动画
            create_route_animation(
                td, 
                actions_trained[i].cpu().numpy(), 
                animation_path,
                title="训练后路线生成过程"
            )
            
            # ========== 保存文件记录到数据库 ==========
            if bg_file_manager:
                try:
                    bg_file_manager.save_file_record(
                        user_id=user_id,
                        session_id=session_id,
                        filename=animation_filename,
                        file_type='animation',
                        file_path=animation_path
                    )
                except Exception as e:
                    print(f"保存文件记录失败: {str(e)}")
            
            animation_paths.append(f"/static/model_plots/user_{user_id}/{animation_filename}")  # ========== 使用用户目录 ==========
        
        # 保存检查点  # 将最终模型权重保存到文件
        trainer.save_checkpoint(checkpoint_path)  # 保存ckpt
        
        # ========== 保存checkpoint文件记录到数据库 ==========
        if bg_file_manager:
            try:
                checkpoint_filename = os.path.basename(checkpoint_path)
                bg_file_manager.save_file_record(
                    user_id=user_id,
                    session_id=session_id,
                    filename=checkpoint_filename,
                    file_type='checkpoint',
                    file_path=checkpoint_path
                )
            except Exception as e:
                print(f"保存checkpoint记录失败: {str(e)}")
        
        queue.put(json.dumps({
            'type': 'info',  # 消息类型
            'message': f'检查点已保存: {checkpoint_path}'  # 保存成功提示
        }))
        
        # 训练完成  # 汇总最终结果并通知前端
        training_status[session_id]['status'] = 'completed'  # 标记状态为已完成
        
        # ========== 更新训练会话状态到数据库 ==========
        if bg_session_manager:
            try:
                from datetime import datetime
                bg_session_manager.update_session(
                    session_id=session_id,
                    status='completed',
                    end_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    final_loss=training_status[session_id]['loss'],
                    final_reward=training_status[session_id]['reward'],
                    best_reward=training_status[session_id]['best_reward'],
                    checkpoint_path=checkpoint_path
                )
            except Exception as e:
                print(f"更新训练会话状态失败: {str(e)}")
        
        final_results = {
            'model': model_type,  # 模型类型
            'problem': problem_type,  # 问题类型
            'strategy': 'REINFORCE',  # 训练策略
            'total_epochs': epochs,  # 总训练轮数
            'final_loss': training_status[session_id]['loss'],  # 最终loss
            'final_reward': training_status[session_id]['reward'],  # 最终reward
            'best_reward': training_status[session_id]['best_reward'],  # 历史最优reward
            'plot_paths': plot_paths,  # 可视化图片路径
            'animation_paths': animation_paths,  # 动态GIF路径
            'training_curve': training_status[session_id].get('plot_url', ''),  # 训练曲线图路径
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
        
        # ========== 更新训练会话状态为失败 ==========
        if bg_session_manager:
            try:
                from datetime import datetime
                bg_session_manager.update_session(
                    session_id=session_id,
                    status='failed',
                    end_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
            except Exception as update_error:
                print(f"更新失败状态失败: {str(update_error)}")
        
        queue.put(json.dumps({  # 向前端推送错误消息
            'type': 'error',  # 消息类型：错误
            'message': f'训练出错: {str(e)}'  # 错误描述
        }))
    
    finally:
        # 关闭后台数据库连接
        if bg_db:
            try:
                bg_db.close()
            except:
                pass


# 模拟训练函数（备用）
def simulate_training(config, session_id, user_id):  # ========== 添加user_id参数 ==========
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
@login_required
def start_training():
    """接收训练配置并启动训练 - 需要登录"""
    try:
        # ========== 获取当前用户ID ==========
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        config = request.json
        
        # 生成唯一的会话 ID
        import uuid
        session_id = str(uuid.uuid4())
        
        # ========== 记录训练会话到数据库 ==========
        session_manager = get_session_manager()
        if session_manager:
            try:
                session_manager.create_session(
                    user_id=user_id,
                    session_id=session_id,
                    model_type=config.get('model', 'attention'),
                    problem_type=config.get('problem', 'tsp'),
                    config=json.dumps(config)
                )
            except Exception as e:
                print(f"记录训练会话失败: {str(e)}")
        
        # 创建消息队列
        training_queues[session_id] = Queue()
        
        # 根据 RL4CO 是否可用选择训练函数
        if RL4CO_AVAILABLE:
            training_func = real_rl4co_training
            mode = "真实训练模式"
        else:
            training_func = simulate_training
            mode = "模拟训练模式"
        
        # 在后台线程中启动训练（传入user_id）
        training_thread = threading.Thread(
            target=training_func,
            args=(config, session_id, user_id),  # ========== 添加user_id参数 ==========
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


@app.route('/api/list_files', methods=['GET'])
@login_required
def list_training_files():
    """列出当前用户的训练产生的文件 - 需要登录"""
    try:
        # ========== 获取当前用户ID，只显示该用户的文件 ==========
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        # ========== 从数据库获取用户文件列表 ==========
        file_manager = get_file_manager()
        if file_manager:
            try:
                user_files = file_manager.get_user_files(user_id)
                storage_stats = file_manager.get_user_storage_stats(user_id)
                
                files_info = {
                    'plots': [],
                    'checkpoints': [],
                    'total_size': storage_stats['total_mb'] if storage_stats else 0,
                    'total_count': storage_stats['total_files'] if storage_stats else 0
                }
                
                for file_record in user_files:
                    file_info = {
                        'id': file_record['id'],
                        'name': file_record['file_name'],
                        'type': file_record['file_type'],
                        'size': file_record['file_size'],
                        'size_mb': round(file_record['file_size'] / (1024 * 1024), 2),
                        'path': f"/static/model_plots/user_{user_id}/{file_record['file_name']}",
                        'session_id': file_record['session_id'],
                        'create_time': file_record['create_time'].strftime('%Y-%m-%d %H:%M')
                    }
                    
                    if file_record['file_type'] in ['plot', 'animation', 'curve']:
                        files_info['plots'].append(file_info)
                    elif file_record['file_type'] == 'checkpoint':
                        files_info['checkpoints'].append(file_info)
                
                return jsonify({
                    'success': True,
                    'files': files_info
                })
                
            except Exception as e:
                print(f"从数据库获取文件失败: {str(e)}")
                # 降级到文件系统扫描
                pass
        
        # ========== 降级方案：直接扫描用户目录 ==========
        files_info = {
            'plots': [],
            'checkpoints': [],
            'total_size': 0
        }
        
        USER_PLOTS_DIR = get_user_plot_dir(user_id)
        USER_CHECKPOINTS_DIR = get_user_checkpoint_dir(user_id)
        
        # 列出可视化图片文件
        if os.path.exists(USER_PLOTS_DIR):
            for filename in os.listdir(USER_PLOTS_DIR):
                file_path = os.path.join(USER_PLOTS_DIR, filename)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    file_type = 'unknown'
                    
                    if filename.startswith('comparison_'):
                        file_type = 'comparison'
                    elif filename.startswith('animation_'):
                        file_type = 'animation'
                    elif filename.startswith('training_curves_'):
                        file_type = 'training_curve'
                    
                    files_info['plots'].append({
                        'name': filename,
                        'type': file_type,
                        'size': file_size,
                        'size_mb': round(file_size / (1024 * 1024), 2),
                        'path': f'/static/model_plots/user_{user_id}/{filename}',
                        'modified': os.path.getmtime(file_path)
                    })
                    files_info['total_size'] += file_size
        
        # 列出检查点文件
        if os.path.exists(CHECKPOINTS_DIR):
            for filename in os.listdir(CHECKPOINTS_DIR):
                file_path = os.path.join(CHECKPOINTS_DIR, filename)
                if os.path.isfile(file_path) and filename.endswith('.ckpt'):
                    file_size = os.path.getsize(file_path)
                    files_info['checkpoints'].append({
                        'name': filename,
                        'type': 'checkpoint',
                        'size': file_size,
                        'size_mb': round(file_size / (1024 * 1024), 2),
                        'path': file_path,
                        'modified': os.path.getmtime(file_path)
                    })
                    files_info['total_size'] += file_size
        
        # 按修改时间降序排序
        files_info['plots'].sort(key=lambda x: x['modified'], reverse=True)
        files_info['checkpoints'].sort(key=lambda x: x['modified'], reverse=True)
        
        files_info['total_size_mb'] = round(files_info['total_size'] / (1024 * 1024), 2)
        files_info['total_count'] = len(files_info['plots']) + len(files_info['checkpoints'])
        
        return jsonify({
            'success': True,
            'files': files_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'获取文件列表失败: {str(e)}'
        }), 500


@app.route('/api/delete_file', methods=['POST'])
@login_required
def delete_training_file():
    """删除指定的训练文件 - 需要登录且只能删除自己的文件"""
    try:
        # ========== 获取当前用户ID ==========
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        data = request.json
        file_id = data.get('file_id')  # ========== 使用file_id而非filename ==========
        filename = data.get('filename')  # 兼容旧版
        
        # ========== 使用数据库方式删除（推荐） ==========
        file_manager = get_file_manager()
        if file_id and file_manager:
            success, message = file_manager.delete_file(file_id, user_id)
            if success:
                return jsonify({
                    'success': True,
                    'message': message
                })
            else:
                return jsonify({
                    'success': False,
                    'message': message
                }), 403
        
        # ========== 降级方案：直接删除文件（兼容） ==========
        if not filename:
            return jsonify({
                'success': False,
                'message': '未提供文件名或文件ID'
            }), 400
        
        file_type = data.get('file_type', 'plot')
        USER_PLOTS_DIR = get_user_plot_dir(user_id)
        USER_CHECKPOINTS_DIR = get_user_checkpoint_dir(user_id)
        
        # 确定文件路径（用户专属目录）
        if file_type == 'checkpoint':
            file_path = os.path.join(USER_CHECKPOINTS_DIR, filename)
        else:
            file_path = os.path.join(USER_PLOTS_DIR, filename)
        
        # 安全检查：确保文件在用户目录内
        abs_file_path = os.path.abspath(file_path)
        abs_user_plots = os.path.abspath(USER_PLOTS_DIR)
        abs_user_checkpoints = os.path.abspath(USER_CHECKPOINTS_DIR)
        
        if not (abs_file_path.startswith(abs_user_plots) or abs_file_path.startswith(abs_user_checkpoints)):
            return jsonify({
                'success': False,
                'message': '无效的文件路径或无权访问'
            }), 403
        
        # 删除文件
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({
                'success': True,
                'message': f'文件 {filename} 已删除'
            })
        else:
            return jsonify({
                'success': False,
                'message': '文件不存在'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'删除文件失败: {str(e)}'
        }), 500


@app.route('/api/download_checkpoint/<filename>')
@login_required
def download_checkpoint(filename):
    """下载检查点文件 - 需要登录且只能下载自己的文件"""
    try:
        # ========== 获取当前用户ID ==========
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        USER_CHECKPOINTS_DIR = get_user_checkpoint_dir(user_id)
        
        # 安全检查：防止路径遍历攻击
        safe_filename = os.path.basename(filename)
        file_path = os.path.join(USER_CHECKPOINTS_DIR, safe_filename)
        
        # 验证文件路径在用户目录内
        abs_file_path = os.path.abspath(file_path)
        abs_user_dir = os.path.abspath(USER_CHECKPOINTS_DIR)
        
        if not abs_file_path.startswith(abs_user_dir):
            return jsonify({
                'success': False,
                'message': '无效的文件路径'
            }), 403
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            return jsonify({
                'success': False,
                'message': '文件不存在'
            }), 404
        
        # 发送文件
        from flask import send_file
        return send_file(
            file_path,
            as_attachment=True,
            download_name=safe_filename,
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'下载失败: {str(e)}'
        }), 500


@app.route('/api/delete_by_session', methods=['POST'])
@login_required
def delete_by_session():
    """根据session_id删除相关的所有文件 - 需要登录且只能删除自己的文件"""
    try:
        # ========== 获取当前用户ID ==========
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({
                'success': False,
                'message': '未提供session_id'
            }), 400
        
        # 取前8位作为文件名前缀
        session_prefix = session_id[:8]
        deleted_files = []
        
        USER_PLOTS_DIR = get_user_plot_dir(user_id)
        
        # 删除用户的可视化文件
        if os.path.exists(USER_PLOTS_DIR):
            for filename in os.listdir(USER_PLOTS_DIR):
                if session_prefix in filename:
                    file_path = os.path.join(USER_PLOTS_DIR, filename)
                    try:
                        os.remove(file_path)
                        deleted_files.append(filename)
                    except Exception as e:
                        print(f"删除文件 {filename} 失败: {str(e)}")
        
        return jsonify({
            'success': True,
            'message': f'已删除 {len(deleted_files)} 个文件',
            'deleted_files': deleted_files
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'批量删除失败: {str(e)}'
        }), 500


@app.route('/api/clear_all_files', methods=['POST'])
@login_required
def clear_all_files():
    """清空当前用户的所有训练文件（谨慎使用）"""
    try:
        # ========== 获取当前用户ID ==========
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({
                'success': False,
                'message': '请先登录'
            }), 401
        
        data = request.json
        confirm = data.get('confirm', False)
        
        if not confirm:
            return jsonify({
                'success': False,
                'message': '需要确认操作'
            }), 400
        
        deleted_count = 0
        
        USER_PLOTS_DIR = get_user_plot_dir(user_id)
        USER_CHECKPOINTS_DIR = get_user_checkpoint_dir(user_id)
        
        # 清空用户的可视化文件
        if os.path.exists(USER_PLOTS_DIR):
            for filename in os.listdir(USER_PLOTS_DIR):
                file_path = os.path.join(USER_PLOTS_DIR, filename)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"删除文件 {filename} 失败: {str(e)}")
        
        # 清空用户的检查点文件
        if os.path.exists(USER_CHECKPOINTS_DIR):
            for filename in os.listdir(USER_CHECKPOINTS_DIR):
                if filename.endswith('.ckpt'):
                    file_path = os.path.join(USER_CHECKPOINTS_DIR, filename)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"删除文件 {filename} 失败: {str(e)}")
        
        # ========== 同时清空数据库记录 ==========
        file_manager = get_file_manager()
        if file_manager:
            try:
                db = get_db()
                cursor = db.cursor()
                cursor.execute("DELETE FROM user_files WHERE user_id = %s", (user_id,))
                db.commit()
            except Exception as e:
                print(f"清空数据库记录失败: {str(e)}")
        
        return jsonify({
            'success': True,
            'message': f'已清空所有文件，共删除 {deleted_count} 个文件'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'清空文件失败: {str(e)}'
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
