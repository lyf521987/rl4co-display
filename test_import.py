#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证app.py是否能正确导入
"""

import sys
import ast

try:
    # 验证语法
    with open('app.py', 'r', encoding='utf-8') as f:
        code = f.read()
    ast.parse(code)
    print("✓ app.py 语法正确")
    
    # 检查关键定义
    if 'class SimpleCache:' in code:
        print("✓ SimpleCache 类已定义")
    
    if 'def cached_api' in code:
        print("✓ cached_api 装饰器已定义")
    
    if '@cached_api(key_prefix=' in code:
        print("✓ cached_api 装饰器已被应用")
        
    # 检查缓存定义在使用之前
    cache_def = code.find('class SimpleCache:')
    cache_use = code.find('@cached_api(key_prefix=')
    
    if cache_def < cache_use:
        print("✓ 缓存装饰器在使用之前已定义")
    else:
        print("✗ 缓存装饰器定义位置错误")
        sys.exit(1)
        
    print("\n✅ 所有检查通过！app.py 文件有效")
    
except SyntaxError as e:
    print(f"✗ 语法错误: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ 错误: {e}")
    sys.exit(1)

