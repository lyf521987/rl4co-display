"""
用户认证模块
RL4CO Display - 用户数据隔离
山西大学 计算机科学与技术学院
"""

from flask import session, request, jsonify, redirect, url_for
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import os


# ============================================
# 1. 登录检查装饰器
# ============================================

def login_required(f):
    """
    登录检查装饰器
    用法：@login_required
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            # API请求返回JSON
            if request.is_json or request.path.startswith('/api/'):
                return jsonify({
                    'success': False, 
                    'message': '请先登录',
                    'redirect': '/login'
                }), 401
            # 页面请求重定向到登录页
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ============================================
# 2. 用户管理类
# ============================================

class UserManager:
    """用户管理类"""
    
    def __init__(self, db):
        self.db = db
    
    def create_user(self, username, password, email=None):
        """
        创建新用户
        
        Args:
            username: 用户名
            password: 密码（明文，会自动加密）
            email: 邮箱（可选）
        
        Returns:
            tuple: (success, message, user_id)
        """
        try:
            # 检查用户名是否已存在
            cursor = self.db.cursor(dictionary=True)
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                return False, '用户名已存在', None
            
            # 加密密码
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            
            # 插入用户
            cursor.execute("""
                INSERT INTO users (username, password, email)
                VALUES (%s, %s, %s)
            """, (username, hashed_password, email))
            
            user_id = cursor.lastrowid
            self.db.commit()
            
            # 创建用户专属目录
            self._create_user_directories(user_id)
            
            return True, '注册成功', user_id
            
        except Exception as e:
            self.db.rollback()
            return False, f'注册失败：{str(e)}', None
    
    def verify_user(self, username, password):
        """
        验证用户登录
        
        Args:
            username: 用户名
            password: 密码（明文）
        
        Returns:
            tuple: (success, message, user_data)
        """
        try:
            cursor = self.db.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            
            if not user:
                return False, '用户不存在', None
            
            if not check_password_hash(user['password'], password):
                return False, '密码错误', None
            
            # 更新最后登录时间
            cursor.execute("""
                UPDATE users SET last_login = NOW() 
                WHERE id = %s
            """, (user['id'],))
            self.db.commit()
            
            return True, '登录成功', {
                'id': user['id'],
                'username': user['username'],
                'email': user['email']
            }
            
        except Exception as e:
            return False, f'登录失败：{str(e)}', None
    
    def get_user(self, user_id):
        """获取用户信息"""
        cursor = self.db.cursor(dictionary=True)
        cursor.execute("SELECT id, username, email, create_time, last_login FROM users WHERE id = %s", (user_id,))
        return cursor.fetchone()
    
    def update_password(self, user_id, old_password, new_password):
        """修改密码"""
        try:
            cursor = self.db.cursor(dictionary=True)
            cursor.execute("SELECT password FROM users WHERE id = %s", (user_id,))
            user = cursor.fetchone()
            
            if not user:
                return False, '用户不存在'
            
            if not check_password_hash(user['password'], old_password):
                return False, '原密码错误'
            
            hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
            cursor.execute("UPDATE users SET password = %s WHERE id = %s", (hashed_password, user_id))
            self.db.commit()
            
            return True, '密码修改成功'
            
        except Exception as e:
            self.db.rollback()
            return False, f'密码修改失败：{str(e)}'
    
    def _create_user_directories(self, user_id):
        """创建用户专属目录"""
        dirs = [
            f'static/model_plots/user_{user_id}',
            f'checkpoints/user_{user_id}'
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


# ============================================
# 3. 训练会话管理类
# ============================================

class TrainingSessionManager:
    """训练会话管理类"""
    
    def __init__(self, db):
        self.db = db
    
    def create_session(self, user_id, session_id, model_type, problem_type, config):
        """创建训练会话记录"""
        try:
            cursor = self.db.cursor()
            cursor.execute("""
                INSERT INTO training_sessions 
                (session_id, user_id, model_type, problem_type, config, status)
                VALUES (%s, %s, %s, %s, %s, 'running')
            """, (session_id, user_id, model_type, problem_type, config))
            self.db.commit()
            return True, '会话创建成功'
        except Exception as e:
            self.db.rollback()
            return False, f'会话创建失败：{str(e)}'
    
    def update_session(self, session_id, **kwargs):
        """更新训练会话"""
        try:
            # 构建更新语句
            update_fields = []
            values = []
            
            allowed_fields = ['status', 'end_time', 'final_loss', 'final_reward', 'best_reward', 'checkpoint_path']
            for key, value in kwargs.items():
                if key in allowed_fields:
                    update_fields.append(f"{key} = %s")
                    values.append(value)
            
            if not update_fields:
                return False, '没有需要更新的字段'
            
            values.append(session_id)
            sql = f"UPDATE training_sessions SET {', '.join(update_fields)} WHERE session_id = %s"
            
            cursor = self.db.cursor()
            cursor.execute(sql, tuple(values))
            self.db.commit()
            
            return True, '会话更新成功'
        except Exception as e:
            self.db.rollback()
            return False, f'会话更新失败：{str(e)}'
    
    def get_user_sessions(self, user_id, limit=50):
        """获取用户的训练会话列表"""
        cursor = self.db.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                ts.*,
                COUNT(tf.id) as file_count,
                COALESCE(SUM(tf.file_size), 0) as total_size
            FROM training_sessions ts
            LEFT JOIN training_files tf ON ts.session_id = tf.session_id
            WHERE ts.user_id = %s
            GROUP BY ts.id
            ORDER BY ts.start_time DESC
            LIMIT %s
        """, (user_id, limit))
        return cursor.fetchall()
    
    def verify_session_owner(self, session_id, user_id):
        """验证训练会话是否属于指定用户"""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT user_id FROM training_sessions 
            WHERE session_id = %s
        """, (session_id,))
        result = cursor.fetchone()
        return result and result[0] == user_id


# ============================================
# 4. 文件管理类
# ============================================

class FileManager:
    """训练文件管理类"""
    
    def __init__(self, db):
        self.db = db
    
    def save_file_record(self, user_id, session_id, filename, file_type, file_path):
        """保存文件记录到数据库"""
        try:
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            
            cursor = self.db.cursor()
            cursor.execute("""
                INSERT INTO training_files 
                (user_id, session_id, file_name, file_type, file_path, file_size)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (user_id, session_id, filename, file_type, file_path, file_size))
            self.db.commit()
            
            return True, '文件记录保存成功'
        except Exception as e:
            self.db.rollback()
            return False, f'文件记录保存失败：{str(e)}'
    
    def get_user_files(self, user_id, file_type=None, session_id=None):
        """获取用户的文件列表"""
        cursor = self.db.cursor(dictionary=True)
        
        sql = "SELECT * FROM training_files WHERE user_id = %s"
        params = [user_id]
        
        if file_type:
            sql += " AND file_type = %s"
            params.append(file_type)
        
        if session_id:
            sql += " AND session_id = %s"
            params.append(session_id)
        
        sql += " ORDER BY create_time DESC"
        
        cursor.execute(sql, tuple(params))
        return cursor.fetchall()
    
    def verify_file_owner(self, file_id, user_id):
        """验证文件是否属于指定用户"""
        cursor = self.db.cursor()
        cursor.execute("SELECT user_id FROM training_files WHERE id = %s", (file_id,))
        result = cursor.fetchone()
        return result and result[0] == user_id
    
    def delete_file(self, file_id, user_id):
        """删除文件（检查权限）"""
        try:
            # 验证权限
            if not self.verify_file_owner(file_id, user_id):
                return False, '无权删除该文件'
            
            # 获取文件路径
            cursor = self.db.cursor(dictionary=True)
            cursor.execute("SELECT file_path FROM training_files WHERE id = %s", (file_id,))
            file_record = cursor.fetchone()
            
            if not file_record:
                return False, '文件记录不存在'
            
            # 删除物理文件
            if os.path.exists(file_record['file_path']):
                os.remove(file_record['file_path'])
            
            # 删除数据库记录
            cursor.execute("DELETE FROM training_files WHERE id = %s", (file_id,))
            self.db.commit()
            
            return True, '文件删除成功'
        except Exception as e:
            self.db.rollback()
            return False, f'文件删除失败：{str(e)}'
    
    def get_user_storage_stats(self, user_id):
        """获取用户存储统计"""
        cursor = self.db.cursor(dictionary=True)
        cursor.execute("""
            SELECT 
                COUNT(*) as total_files,
                COALESCE(SUM(file_size), 0) as total_bytes,
                ROUND(COALESCE(SUM(file_size), 0) / 1024 / 1024, 2) as total_mb,
                COUNT(CASE WHEN file_type = 'plot' THEN 1 END) as plot_count,
                COUNT(CASE WHEN file_type = 'animation' THEN 1 END) as animation_count,
                COUNT(CASE WHEN file_type = 'curve' THEN 1 END) as curve_count,
                COUNT(CASE WHEN file_type = 'checkpoint' THEN 1 END) as checkpoint_count
            FROM training_files
            WHERE user_id = %s
        """, (user_id,))
        return cursor.fetchone()


# ============================================
# 5. 路径安全工具
# ============================================

def get_user_plot_dir(user_id):
    """获取用户专属的图片目录"""
    user_dir = os.path.join('static', 'model_plots', f'user_{user_id}')
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def get_user_checkpoint_dir(user_id):
    """获取用户专属的checkpoint目录"""
    user_dir = os.path.join('checkpoints', f'user_{user_id}')
    os.makedirs(user_dir, exist_ok=True)
    return user_dir


def safe_join_path(base_dir, filename):
    """安全地拼接文件路径，防止路径遍历攻击"""
    # 移除文件名中的危险字符
    safe_name = os.path.basename(filename)
    full_path = os.path.abspath(os.path.join(base_dir, safe_name))
    base_abs = os.path.abspath(base_dir)
    
    # 确保文件在指定目录内
    if not full_path.startswith(base_abs):
        raise ValueError("Invalid file path")
    
    return full_path


# ============================================
# 6. Session工具函数
# ============================================

def get_current_user_id():
    """获取当前登录用户的ID"""
    return session.get('user_id')


def get_current_username():
    """获取当前登录用户的用户名"""
    return session.get('username')


def is_logged_in():
    """检查用户是否已登录"""
    return 'user_id' in session


def set_user_session(user_data):
    """设置用户session"""
    session.permanent = True
    session['user_id'] = user_data['id']
    session['username'] = user_data['username']
    if 'email' in user_data:
        session['email'] = user_data['email']


def clear_user_session():
    """清除用户session"""
    session.clear()


# ============================================
# 示例用法
# ============================================

"""
在 app.py 中使用：

from auth_module import (
    login_required, 
    UserManager, 
    TrainingSessionManager,
    FileManager,
    get_user_plot_dir,
    set_user_session,
    clear_user_session
)

# 初始化管理器
user_manager = UserManager(db)
session_manager = TrainingSessionManager(db)
file_manager = FileManager(db)

# 注册
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    success, message, user_id = user_manager.create_user(
        data['username'], 
        data['password'],
        data.get('email')
    )
    return jsonify({'success': success, 'message': message})

# 登录
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    success, message, user_data = user_manager.verify_user(
        data['username'], 
        data['password']
    )
    if success:
        set_user_session(user_data)
    return jsonify({'success': success, 'message': message, 'user': user_data})

# 受保护的API
@app.route('/api/start_training', methods=['POST'])
@login_required
def start_training():
    user_id = get_current_user_id()
    # ... 训练逻辑
"""

