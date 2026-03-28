import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ===================== 1. 读取数据 =====================
train_df = pd.read_excel("Data4Regression.xlsx", sheet_name=0)
test_df = pd.read_excel("Data4Regression.xlsx", sheet_name=1)

x_train, y_train = train_df.iloc[:, 0].values, train_df.iloc[:, 1].values
x_test, y_test = test_df.iloc[:, 0].values, test_df.iloc[:, 1].values

# 排序（保证拟合曲线平滑）
train_idx = np.argsort(x_train)
x_train_sorted = x_train[train_idx]
y_train_sorted = y_train[train_idx]

# ===================== 2. 核回归实现（高斯核） =====================
def kernel_regression(x_train, y_train, x_pred, h=0.08):  # ✅ 核心修改：h=0.08
    """
    高斯核回归
    x_train: 训练特征
    y_train: 训练标签
    x_pred: 预测点
    h: 带宽（核函数平滑参数）
    """
    y_pred = np.zeros_like(x_pred)
    n = len(x_train)
    for i, x in enumerate(x_pred):
        # 高斯核权重计算
        weights = np.exp(-((x_train - x) ** 2) / (2 * h ** 2))
        weights = weights / np.sum(weights)
        # 加权平均预测
        y_pred[i] = np.sum(weights * y_train)
    return y_pred

# 生成平滑的预测x点
x_plot = np.linspace(0, 10, 500)
# 核回归拟合预测
y_plot = kernel_regression(x_train_sorted, y_train_sorted, x_plot, h=0.08)  # ✅ 修改h
# 训练集/测试集预测
y_train_pred = kernel_regression(x_train, y_train, x_train, h=0.08)  # ✅ 修改h
y_test_pred = kernel_regression(x_train, y_train, x_test, h=0.08)  # ✅ 修改h

# ===================== 3. 计算评估指标（MSE + R²） =====================
# 训练集
train_mse = np.mean((y_train - y_train_pred) ** 2)
ss_res_train = np.sum((y_train - y_train_pred) ** 2)
ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
train_r2 = 1 - (ss_res_train / ss_tot_train)

# 测试集
test_mse = np.mean((y_test - y_test_pred) ** 2)
ss_res_test = np.sum((y_test - y_test_pred) ** 2)
ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
test_r2 = 1 - (ss_res_test / ss_tot_test)

# ===================== 4. 绘制拟合图像 =====================
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color="blue", label="Train Data", s=20, alpha=0.7)
plt.scatter(x_test, y_test, color="green", label="Test Data", s=20, alpha=0.7)
plt.plot(x_plot, y_plot, color="red", linewidth=3, label="Kernel Regression Fit")
plt.title(f"Kernel Regression (Gaussian Kernel)\nTrain MSE={train_mse:.4f}, R²={train_r2:.4f} | Test MSE={test_mse:.4f}, R²={test_r2:.4f}")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ===================== 5. 控制台输出结果 =====================
print("="*70)
print("           Kernel Regression - Fitting Results")
print("="*70)
print(f"Train Set  |  MSE: {train_mse:.6f}  |  R²: {train_r2:.6f}")
print(f"Test Set   |  MSE: {test_mse:.6f}  |  R²: {test_r2:.6f}")
print("="*70)