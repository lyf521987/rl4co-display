"""
TSP动态路线生成演示脚本
演示如何生成逐步构建路线的动态GIF

运行此脚本将生成一个示例TSP问题的动态路线图
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建输出目录
os.makedirs("demo_output", exist_ok=True)


def create_route_animation_demo(num_cities=20, save_path="demo_output/tsp_animation_demo.gif", fps=2):
    """
    创建TSP路线逐步生成的动态GIF演示

    参数:
        num_cities: 城市数量
        save_path: GIF保存路径
        fps: 帧率（每秒帧数）
    """
    print(f"开始生成动态路线图...")
    print(f"城市数量: {num_cities}")
    print(f"保存路径: {save_path}")

    # 生成随机城市坐标
    np.random.seed(42)
    locs = np.random.rand(num_cities, 2)

    # 使用简单的最近邻算法生成路线（模拟训练后的策略）
    actions = nearest_neighbor_route(locs)

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

    print(f"\n生成动画帧...")
    # 为每一步生成一帧图像
    for step in range(num_cities + 1):
        if step % 5 == 0:
            print(f"  进度: {step}/{num_cities}")

        fig, ax = plt.subplots(figsize=(10, 8))

        # 绘制所有城市点（未访问的用浅蓝色，已访问的用深蓝色）
        visited_mask = np.zeros(num_cities, dtype=bool)
        if step > 0:
            for i in range(step):
                visited_mask[actions[i]] = True

        # 未访问的城市
        unvisited = ~visited_mask
        ax.scatter(locs[unvisited, 0], locs[unvisited, 1], c='lightblue', s=200,
                  zorder=3, alpha=0.6, edgecolors='black', linewidths=2,
                  label='未访问')

        # 已访问的城市
        if np.any(visited_mask):
            ax.scatter(locs[visited_mask, 0], locs[visited_mask, 1], c='lightgreen', s=200,
                      zorder=3, alpha=0.8, edgecolors='darkgreen', linewidths=2,
                      label='已访问')

        # 标注城市编号（只标注部分，避免拥挤）
        if num_cities <= 30:
            for i, (x, y) in enumerate(locs):
                ax.text(x, y, str(i), fontsize=9, ha='center', va='center',
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
                       'b-', linewidth=2.5, alpha=0.7, zorder=1)

                # 添加箭头表示方向
                mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
                dx, dy = end[0] - start[0], end[1] - start[1]
                ax.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1),
                          xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                          arrowprops=dict(arrowstyle='->', color='blue',
                                        lw=1.5, alpha=0.7))

        # 高亮当前访问的城市
        if step > 0 and step <= num_cities:
            current_city = actions[step - 1]
            ax.scatter(locs[current_city, 0], locs[current_city, 1],
                      c='red', s=500, zorder=5, marker='*',
                      edgecolors='darkred', linewidths=2,
                      label=f'当前访问')

        # 高亮起点
        start_city = actions[0]
        ax.scatter(locs[start_city, 0], locs[start_city, 1],
                  c='gold', s=350, zorder=4, marker='D',
                  edgecolors='orange', linewidths=2,
                  label='起点')

        # 计算当前累计成本
        current_cost = calculate_partial_distance(locs, actions, step)

        # 设置标题和信息
        if step == 0:
            info_text = ">> 开始构建路线..."
            cost_text = ""
        elif step < num_cities:
            info_text = f"第 {step} 步 | 已访问 {step}/{num_cities} 个城市"
            cost_text = f"累计成本: {current_cost:.3f}"
        else:
            # 最后一步，返回起点
            final_dist = np.sqrt(np.sum((locs[actions[-1]] - locs[actions[0]]) ** 2))
            total_cost = current_cost + final_dist
            info_text = f"【完成】访问了所有 {num_cities} 个城市"
            cost_text = f"总成本: {total_cost:.3f}"

        title = f"TSP 路线生成动态演示\n{info_text}"
        if cost_text:
            title += f"\n{cost_text}"

        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

        # 设置坐标轴
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('X 坐标', fontsize=11)
        ax.set_ylabel('Y 坐标', fontsize=11)

        # 添加图例
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

        # 添加进度条
        progress = step / num_cities
        progress_bar_width = 0.6
        progress_bar_height = 0.03
        progress_bar_x = 0.5 - progress_bar_width / 2
        progress_bar_y = 0.02

        # 绘制进度条背景
        ax.add_patch(plt.Rectangle((progress_bar_x, progress_bar_y), progress_bar_width, progress_bar_height,
                                   transform=ax.transAxes, facecolor='lightgray',
                                   edgecolor='black', linewidth=1, zorder=10))

        # 绘制进度
        if progress > 0:
            ax.add_patch(plt.Rectangle((progress_bar_x, progress_bar_y),
                                      progress_bar_width * progress, progress_bar_height,
                                      transform=ax.transAxes, facecolor='green',
                                      alpha=0.7, zorder=11))

        # 进度文本
        ax.text(0.5, progress_bar_y + progress_bar_height / 2,
               f"{int(progress * 100)}%",
               ha='center', va='center', transform=ax.transAxes,
               fontsize=10, fontweight='bold', color='white' if progress > 0.3 else 'black',
               zorder=12)

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
    print("\n添加结束帧...")
    for _ in range(5):
        frames.append(frames[-1])

    # 保存为GIF
    print(f"\n保存GIF到 {save_path}...")
    duration = int(1000 / fps)  # 每帧持续时间（毫秒）
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=False
    )
    
    print(f"[完成] 动态路线图生成完成！")
    print(f"   文件大小: {os.path.getsize(save_path) / 1024:.2f} KB")
    print(f"   总帧数: {len(frames)}")
    print(f"   帧率: {fps} fps")


def nearest_neighbor_route(locs):
    """
    使用最近邻算法生成TSP路线
    """
    num_cities = len(locs)
    route = [0]  # 从城市0开始
    unvisited = set(range(1, num_cities))

    while unvisited:
        current_city = route[-1]
        current_coords = locs[current_city]

        # 找到距离当前城市最近的未访问城市
        nearest_city = min(
            unvisited,
            key=lambda city: np.sqrt(np.sum((current_coords - locs[city]) ** 2))
        )

        route.append(nearest_city)
        unvisited.remove(nearest_city)

    return np.array(route)


def calculate_route_cost(locs, route):
    """计算路线的总成本"""
    total_cost = 0.0
    for i in range(len(route)):
        start = locs[route[i]]
        end = locs[route[(i + 1) % len(route)]]
        total_cost += np.sqrt(np.sum((start - end) ** 2))
    return total_cost


if __name__ == "__main__":
    print("=" * 80)
    print("TSP 动态路线生成演示")
    print("=" * 80)

    # 生成20个城市的动态路线图
    create_route_animation_demo(
        num_cities=20,
        save_path="demo_output/tsp_animation_20cities.gif",
        fps=2
    )

    print("\n" + "=" * 80)
    print("你也可以生成不同规模的问题：")
    print("  - 10个城市（快速）")
    print("  - 30个城市（中等）")
    print("  - 50个城市（与实际训练相同）")
    print("=" * 80)

    # 可选：生成更多示例
    choice = input("\n是否生成更多示例？(y/n): ").lower()
    if choice == 'y':
        print("\n生成10个城市的示例...")
        create_route_animation_demo(
            num_cities=10,
            save_path="demo_output/tsp_animation_10cities.gif",
            fps=3
        )

        print("\n生成50个城市的示例（与实际训练相同）...")
        create_route_animation_demo(
            num_cities=50,
            save_path="demo_output/tsp_animation_50cities.gif",
            fps=5
        )
        
        print("\n[完成] 所有示例生成完成！")
        print(f"   请查看 demo_output/ 目录")

    print("\n" + "=" * 80)
    print("提示：")
    print("  - 在浏览器中打开GIF文件即可查看动画")
    print("  - 动画展示了路线是如何一步一步构建的")
    print("  - 红色星星表示当前正在访问的城市")
    print("  - 绿色表示已访问，蓝色表示未访问")
    print("  - 进度条显示完成百分比")
    print("=" * 80)
