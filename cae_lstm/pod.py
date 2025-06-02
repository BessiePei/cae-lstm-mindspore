import numpy as np
import matplotlib.pyplot as plt

data = np.load('./dataset/riemann.npy')
reshaped_data = data.reshape(1250, -1)  # (120, 16384)
U,s,V =np.linalg.svd(reshaped_data, full_matrices=False) # SVD就实现了POD分解  U(1250,1250) s(1250,) V(1250,16384)

num_modes = 10      # 此处为保留模态个数
U_modes = U[:, :num_modes]
S_modes = s[:num_modes]
V_modes = V[:num_modes, :]

# 计算中间状态 B (模态系数)
B_modes = np.dot(U_modes, np.diag(S_modes))   # 特征矩阵

# 使用 B 和 Vt 进行重构
reconstructed_data = np.dot(B_modes, V_modes)

reconstructed_data = reconstructed_data.reshape(-1, 128, 128)

# 单帧流场可视化
plt.figure(figsize=(6, 4))
plt.imshow(reconstructed_data[1249], extent=(0, 1.0, 0, 1.0), cmap='jet', origin='upper',
               vmin=np.percentile(data, 0),
               vmax=np.percentile(data, 100))  # 显式设置坐标范围
plt.colorbar(label='Density Distribution')
plt.title('Flow Field at t=0.25s')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./prediction-10/vis/pod_1250.png')
plt.show()

# 登高线图可视化
plt.figure(figsize=(5, 4))
contour = plt.contourf(reconstructed_data[1249], extent=(0, 1.0, 0, 1.0), levels=10, origin='upper', vmin=np.percentile(data, 0), vmax=np.percentile(data, 100), cmap='jet')  # 填充等高线
plt.contour(data[1249], extent=(0, 1.0, 0, 1.0), levels=10, origin='upper', vmin=np.percentile(data, 0), vmax=np.percentile(data, 100), colors='black', linewidths=0.5)  # 添加等高线的边界
plt.xticks(np.linspace(0, 1, 6))  # 横轴显示 0 到 1 的刻度值，包括端点
plt.yticks(np.linspace(0, 1, 6))  # 纵轴显示 0 到 1 的刻度值，包括端点
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('./prediction-10/vis/pod_1250_h.png')
plt.show()


