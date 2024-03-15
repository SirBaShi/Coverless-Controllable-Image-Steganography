import numpy as np
import cv2
import math

img = cv2.imread('7.jpg')
noise = np.random.normal(0,1.5,img.shape)
noisy_image = img+noise.astype(np.uint8)
cv2.imwrite('noisy_img.jpg',noisy_image)

for c in range(3):
    x0 = noisy_image[:, :, c] / 255.0
    x1 = img[:, :, c] / 255.0
    Q = 0.005 # 图像的平滑程度
    gamma = 0.95 # 图像的相邻像素的相似度的权重和核参数
    x_r = np.ones_like(x1) # 参考向量
    pi = 0.5 # 混合系数
    mu = np.mean(x1-x0) # 噪声或异常值的均值
    sigma = np.std(x1-x0) # 噪声或异常值的方差
    alpha = 0.99 # 观测噪声协方差矩阵的参数
    beta = 0.9 # 观测噪声协方差矩阵的参数
    delta = 0.7 # 观测噪声协方差矩阵的参数

    P0 = np.eye(x0.shape[0], x0.shape[1])

    # 定义最大迭代次数和收敛阈值
    max_iter = 10
    tol = 1e-6

    x = x0
    P = P0

    for k in range(max_iter):
        x_pred = x # 状态预测
        P_pred = P + Q # 协方差预测

        h = np.exp(-gamma * np.linalg.norm(x_pred - x_r, axis=1)**2)

        H = -2 * gamma * (x_pred - x_r) * h

        hist, _ = np.histogram(x_pred, bins=256, range=(0, 1), density=True)
        S = np.sum(hist * np.log(hist + 1e-9))

        C = np.zeros_like(x_pred)
        for i in range(x_pred.shape[0]):
            for j in range(x_pred.shape[1]):
                N = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                N = [(i, j) for (i, j) in N if 0 <= i < x_pred.shape[0] and 0 <= j < x_pred.shape[1]]
                N = tuple(N)
                C[i, j] = np.mean((x_pred[i, j] - x_pred[N[0]])**2)

        R = alpha * math.log(e, 1 + beta * S) + delta * C

        K = P_pred * H / (H * P_pred * H + R)

        img = img.astype(np.float32)
        x = x_pred + K * (img[:, :, c] - h)

        P = (1 - K * H) * P_pred

        if np.linalg.norm(x - x_pred) < tol:
            break

    img[:, :, c] = (x * 255).astype(np.uint8)

cv2.imwrite('image_denoised' + str(Q) + str(gamma) + '.jpg', img)