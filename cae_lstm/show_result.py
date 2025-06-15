import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = np.load('./dataset/riemann.npy')


# 检查数据特性
print(f"数据维度: {data.shape}")
print(f"时间步数: {data.shape[0]}")
print(f"空间分辨率: {data.shape[1]}x{data.shape[2]}")
print(f"数值范围: {np.min(data):.2f} ~ {np.max(data):.2f}")


# 单帧流场可视化
plt.figure(figsize=(6, 4))
plt.imshow(data[1099], extent=(0, 1.0, 0, 1.0), cmap='jet', origin='upper',
               vmin=np.percentile(data, 0),
               vmax=np.percentile(data, 100))  # 显式设置坐标范围
plt.colorbar(label='Density Distribution')
plt.title('Flow Field at t=0.25s')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./prediction/vis/ori_data_1250.png')
plt.show()

# 等高线图
plt.figure(figsize=(5, 4))
contour = plt.contourf(data[1099], extent=(0, 1.0, 0, 1.0), levels=10, origin='upper', vmin=np.percentile(data, 0), vmax=np.percentile(data, 100), cmap='jet')  # 填充等高线
plt.contour(data[1099], extent=(0, 1.0, 0, 1.0), levels=10, origin='upper', vmin=np.percentile(data, 0), vmax=np.percentile(data, 100), colors='black', linewidths=0.5)  # 添加等高线的边界
plt.xticks(np.linspace(0, 1, 6))  # 横轴显示 0 到 1 的刻度值，包括端点
plt.yticks(np.linspace(0, 1, 6))  # 纵轴显示 0 到 1 的刻度值，包括端点
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./prediction/vis/ori_data_1250_h.png')
plt.show()

# # 创建画布
# fig, ax = plt.subplots(figsize=(6, 4))
# img = ax.imshow(data[0], extent=(0, 1.0, 0, 1.0), cmap='jet', origin='upper',
#                vmin=np.percentile(data, 0),
#                vmax=np.percentile(data, 100))
# plt.colorbar(img, label='Density Distribution')
# time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')
#
#
# def update(frame):
#     img.set_array(data[frame])
#     time_text.set_text(f'Time Step: {frame}/{len(data)-1}')
#     return img, time_text
#
#
# ani = FuncAnimation(fig, update, frames=len(data),
#                    interval=50, blit=True)
# plt.show()
#
# # 保存动画（可选）
# ani.save('./summary/vis/ori_data_flow_animation.gif', writer='pillow', dpi=300)


