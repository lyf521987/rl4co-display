"""
TSP路线生成过程示例代码
展示如何从城市坐标到最终路线的完整流程
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# 第一步：准备TSP问题实例
# ============================================================================

class TSPInstance:
    """TSP问题实例"""
    def __init__(self, num_cities=10):
        # 生成随机城市坐标 (为了演示，使用较少的城市数)
        self.num_cities = num_cities
        self.city_coords = np.random.rand(num_cities, 2)  # 10个城市，每个2维坐标
        
    def calculate_distance(self, city_a, city_b):
        """计算两个城市之间的欧几里得距离"""
        return np.sqrt(np.sum((city_a - city_b) ** 2))
    
    def calculate_route_cost(self, route):
        """计算路线的总成本（总距离）"""
        total_distance = 0
        for i in range(len(route) - 1):
            city_a = self.city_coords[route[i]]
            city_b = self.city_coords[route[i+1]]
            total_distance += self.calculate_distance(city_a, city_b)
        
        # 加上从最后一个城市返回起点的距离
        total_distance += self.calculate_distance(
            self.city_coords[route[-1]], 
            self.city_coords[route[0]]
        )
        return total_distance


# ============================================================================
# 第二步：策略网络的简化模拟
# ============================================================================

class SimplifiedAttentionPolicy:
    """
    简化版的注意力策略网络（用于演示原理）
    真实的RL4CO使用更复杂的Transformer架构
    """
    def __init__(self, tsp_instance):
        self.tsp = tsp_instance
        
    def random_policy(self):
        """
        随机策略（Sampling解码）
        模拟未训练的模型行为
        """
        route = list(range(self.tsp.num_cities))
        np.random.shuffle(route)  # 随机打乱访问顺序
        return route
    
    def nearest_neighbor_policy(self):
        """
        最近邻策略（模拟训练后的Greedy解码）
        这是一个简单的启发式算法，真实的神经网络策略更智能
        """
        route = [0]  # 从城市0开始
        unvisited = set(range(1, self.tsp.num_cities))
        
        while unvisited:
            current_city = route[-1]
            current_coords = self.tsp.city_coords[current_city]
            
            # 找到距离当前城市最近的未访问城市
            nearest_city = min(
                unvisited, 
                key=lambda city: self.tsp.calculate_distance(
                    current_coords, 
                    self.tsp.city_coords[city]
                )
            )
            
            route.append(nearest_city)
            unvisited.remove(nearest_city)
        
        return route


# ============================================================================
# 第三步：可视化路线
# ============================================================================

def visualize_route(tsp_instance, route, title, ax, color='blue'):
    """可视化TSP路线"""
    coords = tsp_instance.city_coords
    
    # 绘制城市点
    ax.scatter(coords[:, 0], coords[:, 1], c='red', s=100, zorder=3, label='城市')
    
    # 标注城市编号
    for i, (x, y) in enumerate(coords):
        ax.annotate(str(i), (x, y), fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle='circle', facecolor='white', edgecolor='red'))
    
    # 绘制路线
    for i in range(len(route)):
        start = coords[route[i]]
        end = coords[route[(i + 1) % len(route)]]  # 最后一个连回起点
        
        ax.plot([start[0], end[0]], [start[1], end[1]], 
               c=color, linewidth=2, alpha=0.6, zorder=1)
        
        # 添加箭头表示方向
        mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        dx, dy = end[0] - start[0], end[1] - start[1]
        ax.arrow(mid_x - dx*0.1, mid_y - dy*0.1, dx*0.2, dy*0.2,
                head_width=0.03, head_length=0.02, fc=color, ec=color, alpha=0.6)
    
    # 高亮起点
    ax.scatter(coords[0, 0], coords[0, 1], c='green', s=200, 
              marker='*', zorder=4, label='起点')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()


# ============================================================================
# 第四步：完整演示
# ============================================================================

def demonstrate_tsp_route_generation():
    """演示TSP路线生成的完整过程"""
    
    print("=" * 80)
    print("TSP路线生成演示")
    print("=" * 80)
    
    # 1. 创建TSP实例
    print("\n[步骤1] 创建TSP问题实例...")
    tsp = TSPInstance(num_cities=10)
    print(f"生成了 {tsp.num_cities} 个随机城市")
    print(f"城市坐标示例：")
    for i in range(min(3, tsp.num_cities)):
        print(f"  城市 {i}: ({tsp.city_coords[i, 0]:.3f}, {tsp.city_coords[i, 1]:.3f})")
    
    # 2. 初始化策略
    print("\n[步骤2] 初始化策略网络...")
    policy = SimplifiedAttentionPolicy(tsp)
    print("策略网络准备完成")
    
    # 3. 生成随机策略路线（模拟未训练）
    print("\n[步骤3] 生成随机策略路线（Sampling解码）...")
    random_route = policy.random_policy()
    random_cost = tsp.calculate_route_cost(random_route)
    print(f"随机路线: {random_route}")
    print(f"随机路线总成本: {random_cost:.4f}")
    
    # 4. 生成训练后策略路线（模拟训练后）
    print("\n[步骤4] 生成训练后策略路线（Greedy解码）...")
    trained_route = policy.nearest_neighbor_policy()
    trained_cost = tsp.calculate_route_cost(trained_route)
    print(f"优化路线: {trained_route}")
    print(f"优化路线总成本: {trained_cost:.4f}")
    
    # 5. 计算改进
    improvement = ((random_cost - trained_cost) / random_cost) * 100
    print(f"\n[结果] 路线改进: {improvement:.2f}%")
    print(f"  - 随机策略成本: {random_cost:.4f}")
    print(f"  - 训练后成本: {trained_cost:.4f}")
    print(f"  - 节省距离: {random_cost - trained_cost:.4f}")
    
    # 6. 可视化对比
    print("\n[步骤5] 生成可视化对比图...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    visualize_route(tsp, random_route, 
                   f"Random Policy | Cost = {random_cost:.3f}", 
                   ax1, color='red')
    
    visualize_route(tsp, trained_route, 
                   f"Trained Policy | Cost = {trained_cost:.3f}", 
                   ax2, color='blue')
    
    plt.tight_layout()
    plt.savefig('tsp_route_comparison_demo.png', dpi=150, bbox_inches='tight')
    print("可视化图片已保存: tsp_route_comparison_demo.png")
    
    # plt.show()  # 如果需要显示图片，取消注释
    
    print("\n" + "=" * 80)
    print("演示完成！")
    print("=" * 80)


# ============================================================================
# 第五步：详细的逐步解码过程演示
# ============================================================================

def demonstrate_step_by_step_decoding():
    """演示逐步解码过程"""
    print("\n\n" + "=" * 80)
    print("逐步解码过程演示（展示策略网络如何逐个选择城市）")
    print("=" * 80)
    
    tsp = TSPInstance(num_cities=5)  # 使用更少城市便于展示
    
    print(f"\n城市坐标：")
    for i in range(tsp.num_cities):
        print(f"  城市 {i}: ({tsp.city_coords[i, 0]:.3f}, {tsp.city_coords[i, 1]:.3f})")
    
    print("\n" + "-" * 80)
    print("开始逐步构建路线（Greedy策略 - 最近邻）")
    print("-" * 80)
    
    route = [0]  # 从城市0开始
    unvisited = set(range(1, tsp.num_cities))
    
    step = 1
    while unvisited:
        current_city = route[-1]
        current_coords = tsp.city_coords[current_city]
        
        print(f"\n第 {step} 步：")
        print(f"  当前位置: 城市 {current_city} - ({current_coords[0]:.3f}, {current_coords[1]:.3f})")
        print(f"  已访问: {route}")
        print(f"  未访问: {sorted(list(unvisited))}")
        
        # 计算到所有未访问城市的距离
        print(f"  到各未访问城市的距离：")
        distances = {}
        for city in unvisited:
            dist = tsp.calculate_distance(current_coords, tsp.city_coords[city])
            distances[city] = dist
            print(f"    → 城市 {city}: {dist:.4f}")
        
        # 选择最近的城市
        nearest_city = min(distances, key=distances.get)
        print(f"  ✓ 选择城市 {nearest_city}（距离最近: {distances[nearest_city]:.4f}）")
        
        route.append(nearest_city)
        unvisited.remove(nearest_city)
        step += 1
    
    # 返回起点
    print(f"\n第 {step} 步：")
    print(f"  所有城市已访问完毕，返回起点")
    final_dist = tsp.calculate_distance(
        tsp.city_coords[route[-1]], 
        tsp.city_coords[route[0]]
    )
    print(f"  从城市 {route[-1]} 返回城市 {route[0]}: {final_dist:.4f}")
    
    total_cost = tsp.calculate_route_cost(route)
    print(f"\n完整路线: {route} → 0")
    print(f"总成本: {total_cost:.4f}")
    
    print("\n" + "=" * 80)


# ============================================================================
# 运行演示
# ============================================================================

if __name__ == "__main__":
    # 设置随机种子以获得可重复的结果
    np.random.seed(42)
    
    # 运行主演示
    demonstrate_tsp_route_generation()
    
    # 运行逐步解码演示
    demonstrate_step_by_step_decoding()
    
    print("\n\n" + "=" * 80)
    print("注意事项：")
    print("=" * 80)
    print("1. 本演示使用简化的最近邻算法模拟训练后的策略")
    print("2. 真实的RL4CO使用Attention Model，能学习更复杂的模式")
    print("3. 神经网络策略通过REINFORCE算法训练，不是简单的启发式")
    print("4. 实际应用中使用50个城市，本演示使用10个以便观察")
    print("5. Greedy解码在神经网络中是选择概率最高的动作，不是最近邻")
    print("=" * 80)

