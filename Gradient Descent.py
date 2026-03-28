import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 读取Excel数据
train_df = pd.read_excel("Data4Regression.xlsx", sheet_name=0)
test_df = pd.read_excel("Data4Regression.xlsx", sheet_name=1)

x_train, y_train = train_df.iloc[:,0].values, train_df.iloc[:,1].values
x_test, y_test = test_df.iloc[:,0].values, test_df.iloc[:,1].values

X_train = np.c_[np.ones_like(x_train), x_train]
X_test = np.c_[np.ones_like(x_test), x_test]

# 梯度下降超参数设置
lr = 0.001        # 学习率
epochs = 50000    # 最大迭代次数
tol = 1e-8        # 收敛阈值
theta = np.zeros(2)  # 初始化参数 [b, w]
loss_history = []  # 记录损失变化

# 梯度下降迭代优化
for i in range(epochs):
    y_pred = X_train @ theta
    # 计算梯度
    grad = X_train.T @ (y_pred - y_train) / len(y_train)
    # 更新参数
    theta -= lr * grad
    # 记录MSE损失
    loss = np.mean((y_train - y_pred) ** 2)
    loss_history.append(loss)
    # 收敛判断：损失变化小于阈值则停止
    if i > 0 and abs(loss_history[-1] - loss_history[-2]) < tol:
        break

b, w = theta[0], theta[1]

# 预测值
y_train_pred = X_train @ theta
y_test_pred = X_test @ theta

# 计算MSE
train_mse = np.mean((y_train - y_train_pred) ** 2)
test_mse = np.mean((y_test - y_test_pred) ** 2)

# 计算R²【新增核心代码】
ss_res_train = np.sum((y_train - y_train_pred) ** 2)
ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
r2_train = 1 - (ss_res_train / ss_tot_train)

ss_res_test = np.sum((y_test - y_test_pred) ** 2)
ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
r2_test = 1 - (ss_res_test / ss_tot_test)

# 绘制双图：拟合结果 + 损失收敛曲线
plt.figure(figsize=(12, 5))
# 子图1：拟合直线
plt.subplot(1, 2, 1)
plt.scatter(x_train, y_train, color="blue", label="Train Data", s=15, alpha=0.6)
plt.scatter(x_test, y_test, color="green", label="Test Data", s=15, alpha=0.6)
x_line = np.linspace(0, 10, 100)
plt.plot(x_line, b + w*x_line, color="red", linewidth=2, label="Fit Line")
plt.title(f"Gradient Descent\nTrain MSE={train_mse:.4f}, R²={r2_train:.4f} | Test MSE={test_mse:.4f}, R²={r2_test:.4f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(alpha=0.3)

# 子图2：损失收敛曲线
plt.subplot(1, 2, 2)
plt.plot(loss_history, color="orange", linewidth=1.5)
plt.title(f"Loss Convergence | Total Iter: {len(loss_history)}")
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# 控制台输出详细结果
print("="*60)
print("Gradient Descent - Final Result")
print("="*60)
print(f"Intercept (b) = {b:.6f}")
print(f"Slope      (w) = {w:.6f}")
print("-"*60)
print(f"Train Set - MSE: {train_mse:.6f}, R²: {r2_train:.6f}")
print(f"Test Set  - MSE: {test_mse:.6f}, R²: {r2_test:.6f}")
print("-"*60)
print(f"Total Iterations: {len(loss_history)}")
print("="*60)