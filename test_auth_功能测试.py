"""
用户认证功能测试脚本
测试用户注册、登录、数据隔离等功能

使用方法：
1. 启动Flask应用：python app.py
2. 运行此脚本：python test_auth_功能测试.py
"""

import requests
import json
from datetime import datetime

# 配置
BASE_URL = "http://localhost:5000"
session1 = requests.Session()
session2 = requests.Session()

# 颜色输出
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.END}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.END}")

def test_register(username, password, session):
    """测试用户注册"""
    print_info(f"测试注册用户: {username}")
    
    response = session.post(
        f"{BASE_URL}/api/register",
        json={
            "username": username,
            "password": password,
            "email": f"{username}@test.com"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            print_success(f"用户 {username} 注册成功")
            return True
        else:
            print_warning(f"注册失败: {result.get('message')}")
            return False
    else:
        print_error(f"注册失败，HTTP状态码: {response.status_code}")
        return False

def test_login(username, password, session):
    """测试用户登录"""
    print_info(f"测试登录用户: {username}")
    
    response = session.post(
        f"{BASE_URL}/api/login",
        json={
            "username": username,
            "password": password
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            print_success(f"用户 {username} 登录成功")
            return True
        else:
            print_error(f"登录失败: {result.get('message')}")
            return False
    else:
        print_error(f"登录失败，HTTP状态码: {response.status_code}")
        return False

def test_current_user(session, expected_username=None):
    """测试获取当前用户信息"""
    print_info("测试获取当前用户信息")
    
    response = session.get(f"{BASE_URL}/api/current_user")
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            username = result.get('user', {}).get('username')
            print_success(f"当前登录用户: {username}")
            
            if expected_username and username == expected_username:
                print_success(f"用户名匹配: {expected_username}")
                return True
            elif expected_username:
                print_error(f"用户名不匹配，期望: {expected_username}, 实际: {username}")
                return False
            return True
        else:
            print_warning("未登录")
            return False
    else:
        print_error(f"请求失败，HTTP状态码: {response.status_code}")
        return False

def test_start_training(session, model="attention", problem="tsp"):
    """测试开始训练"""
    print_info(f"测试开始训练: {model} - {problem}")
    
    response = session.post(
        f"{BASE_URL}/api/start_training",
        json={
            "model": model,
            "problem": problem,
            "epochs": 2,
            "batch_size": 16
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            session_id = result.get('session_id')
            print_success(f"训练开始成功，session_id: {session_id}")
            return session_id
        else:
            print_error(f"训练开始失败: {result.get('message')}")
            return None
    else:
        print_error(f"请求失败，HTTP状态码: {response.status_code}")
        return None

def test_list_files(session):
    """测试获取文件列表"""
    print_info("测试获取文件列表")
    
    response = session.get(f"{BASE_URL}/api/list_files")
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            files = result.get('files', {})
            plot_count = len(files.get('plots', []))
            checkpoint_count = len(files.get('checkpoints', []))
            total_size = files.get('total_size_mb', 0)
            
            print_success(f"文件列表获取成功:")
            print(f"  - 图片文件: {plot_count} 个")
            print(f"  - Checkpoint文件: {checkpoint_count} 个")
            print(f"  - 总大小: {total_size} MB")
            return plot_count + checkpoint_count
        else:
            print_error(f"获取文件列表失败: {result.get('message')}")
            return 0
    else:
        print_error(f"请求失败，HTTP状态码: {response.status_code}")
        return 0

def test_logout(session):
    """测试登出"""
    print_info("测试登出")
    
    response = session.post(f"{BASE_URL}/api/logout")
    
    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            print_success("登出成功")
            return True
        else:
            print_error(f"登出失败: {result.get('message')}")
            return False
    else:
        print_error(f"请求失败，HTTP状态码: {response.status_code}")
        return False

def test_unauthorized_access(session):
    """测试未登录访问"""
    print_info("测试未登录访问受保护的API")
    
    response = session.get(f"{BASE_URL}/api/list_files")
    
    if response.status_code == 401:
        print_success("未登录访问被正确拦截（返回401）")
        return True
    elif response.status_code == 200:
        print_error("安全漏洞：未登录用户可以访问受保护的API")
        return False
    else:
        print_warning(f"意外的状态码: {response.status_code}")
        return False

def main():
    """主测试流程"""
    print("\n" + "="*60)
    print("用户认证功能测试 - RL4CO Display")
    print("="*60 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user1 = f"testuser1_{timestamp}"
    user2 = f"testuser2_{timestamp}"
    password = "test123456"
    
    results = {}
    
    # ====================================
    # 测试1: 用户注册
    # ====================================
    print("\n【测试1】用户注册功能")
    print("-" * 60)
    results['register_user1'] = test_register(user1, password, session1)
    results['register_user2'] = test_register(user2, password, session2)
    
    # ====================================
    # 测试2: 用户登录
    # ====================================
    print("\n【测试2】用户登录功能")
    print("-" * 60)
    results['login_user1'] = test_login(user1, password, session1)
    results['login_user2'] = test_login(user2, password, session2)
    
    # ====================================
    # 测试3: 获取当前用户信息
    # ====================================
    print("\n【测试3】获取当前用户信息")
    print("-" * 60)
    results['current_user1'] = test_current_user(session1, user1)
    results['current_user2'] = test_current_user(session2, user2)
    
    # ====================================
    # 测试4: 数据隔离 - 用户1训练并查看文件
    # ====================================
    print("\n【测试4】数据隔离测试 - 用户1")
    print("-" * 60)
    print_info("用户1训练模型...")
    # 注意：实际测试时可能需要等待训练完成
    # 这里只测试API调用是否成功
    results['start_training_user1'] = test_start_training(session1, "attention", "tsp")
    
    print_info("等待5秒...")
    import time
    time.sleep(5)
    
    print_info("用户1查看文件列表...")
    file_count_user1 = test_list_files(session1)
    
    # ====================================
    # 测试5: 数据隔离 - 用户2查看文件
    # ====================================
    print("\n【测试5】数据隔离测试 - 用户2")
    print("-" * 60)
    print_info("用户2查看文件列表（应该看不到用户1的文件）...")
    file_count_user2 = test_list_files(session2)
    
    if file_count_user2 == 0:
        print_success("数据隔离成功：用户2看不到用户1的文件")
        results['data_isolation'] = True
    else:
        print_error(f"数据隔离失败：用户2能看到 {file_count_user2} 个文件")
        results['data_isolation'] = False
    
    # ====================================
    # 测试6: 登出功能
    # ====================================
    print("\n【测试6】用户登出功能")
    print("-" * 60)
    results['logout_user1'] = test_logout(session1)
    
    # ====================================
    # 测试7: 未登录访问
    # ====================================
    print("\n【测试7】未登录访问测试")
    print("-" * 60)
    # 创建新session（未登录）
    session3 = requests.Session()
    results['unauthorized_access'] = test_unauthorized_access(session3)
    
    # ====================================
    # 测试总结
    # ====================================
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    failed_tests = total_tests - passed_tests
    
    print(f"\n总测试数: {total_tests}")
    print(f"通过: {passed_tests} {Colors.GREEN}✓{Colors.END}")
    print(f"失败: {failed_tests} {Colors.RED}✗{Colors.END}")
    print(f"通过率: {passed_tests/total_tests*100:.1f}%\n")
    
    # 详细结果
    print("详细结果:")
    for test_name, passed in results.items():
        status = f"{Colors.GREEN}通过{Colors.END}" if passed else f"{Colors.RED}失败{Colors.END}"
        print(f"  {test_name}: {status}")
    
    print("\n" + "="*60)
    
    if failed_tests == 0:
        print_success("所有测试通过！用户认证功能正常工作。")
    else:
        print_warning(f"有 {failed_tests} 个测试失败，请检查相关功能。")
    
    print("\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
    except Exception as e:
        print_error(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

