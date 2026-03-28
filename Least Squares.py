import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 读取Excel数据（索引读取，兼容任意表名）
train_df = pd.read_excel("Data4Regression.xlsx", sheet_name=0)
test_df = pd.read_excel("Data4Regression.xlsx", sheet_name=1)

# 提取特征和标签（第一列x，第二列y）
x_train, y_train = train_df.iloc[:,0].values, train_df.iloc[:,1].values
x_test, y_test = test_df.iloc[:,0].values, test_df.iloc[:,1].values

# 构造带截距项的矩阵 X = [1, x]
X_train = np.c_[np.ones_like(x_train), x_train]
X_test = np.c_[np.ones_like(x_test), x_test]

# 最小二乘法求解参数 [b, w]
theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
b, w = theta[0], theta[1]

# 预测值
y_train_pred = X_train @ theta
y_test_pred = X_test @ theta

# 计算MSE（均方误差）
train_mse = np.mean((y_train - y_train_pred) ** 2)
test_mse = np.mean((y_test - y_test_pred) ** 2)

# 计算R²（决定系数）【新增核心代码】
ss_res_train = np.sum((y_train - y_train_pred) ** 2)  # 残差平方和
ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)  # 总平方和
r2_train = 1 - (ss_res_train / ss_tot_train)

ss_res_test = np.sum((y_test - y_test_pred) ** 2)
ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
r2_test = 1 - (ss_res_test / ss_tot_test)

# 绘制拟合图像（含MSE+R²）
plt.figure(figsize=(10, 5))
plt.scatter(x_train, y_train, color="blue", label="Train Data", s=15, alpha=0.6)
plt.scatter(x_test, y_test, color="green", label="Test Data", s=15, alpha=0.6)
# 拟合直线
x_line = np.linspace(0, 10, 100)
y_line = b + w * x_line
plt.plot(x_line, y_line, color="red", linewidth=2, label=f"Fit Line: y={w:.4f}x{b:.4f}")
# 标题展示双指标
plt.title(f"Least Squares\nTrain MSE={train_mse:.4f}, R²={r2_train:.4f} | Test MSE={test_mse:.4f}, R²={r2_test:.4f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 控制台输出详细结果（含MSE+R²）
print("="*60)
print("Least Squares - Final Result")
print("="*60)
print(f"Intercept (b) = {b:.6f}")
print(f"Slope      (w) = {w:.6f}")
print("-"*60)
print(f"Train Set - MSE: {train_mse:.6f}, R²: {r2_train:.6f}")
print(f"Test Set  - MSE: {test_mse:.6f}, R²: {r2_test:.6f}")
print("="*60)