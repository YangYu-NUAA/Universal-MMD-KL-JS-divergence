import torch
import numpy as np

def gaussian_js(mu1, sigma1, mu2, sigma2):
    # calculate mean and covariance of the two distributions
    mu = (mu1 + mu2) / 2.0
    cov1 = np.diag(sigma1 ** 2)
    cov2 = np.diag(sigma2 ** 2)
    cov = (cov1 + cov2) / 2.0

    #Here can obtain the KL divergence
    print("klsource",kl_divergence(mu1, cov1, mu, cov))
    print("kltarget",kl_divergence(mu2, cov2, mu, cov))


    # calculate JS divergence
    js = (np.sum((mu1 - mu)**2) + np.sum((mu2 - mu)**2)) / 8.0 \
         + 0.5 * kl_divergence(mu1, cov1, mu, cov) \
         + 0.5 * kl_divergence(mu2, cov2, mu, cov)
         
    return js

def kl_divergence(mu1, cov1, mu2, cov2):
    # calculate KL divergence between two multivariate Gaussian distributions
    n = len(mu1)

    
    icov2 = np.linalg.inv(cov2)
    diff = mu2 - mu1
    val = np.trace(icov2.dot(cov1)) \
          + diff.T.dot(icov2).dot(diff) \
          - n + np.log(np.linalg.det(cov2)/np.linalg.det(cov1))
          
    return val / 2.0

def gaussian_kernel(X, Y, sigma=1.0):
    """
    计算高斯核矩阵
    X: shape (n_samples_1, n_features)
    Y: shape (n_samples_2, n_features)
    sigma: 核宽度参数
    """
    n = X.shape[0]
    m = Y.shape[0]
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)
    rx = np.diag(XX).reshape((n, 1)).repeat(m, axis=1)
    ry = np.diag(YY).reshape((m, 1)).repeat(n, axis=1)
    K = np.exp(-(rx.T + ry - 2 * XY) / (2 * sigma ** 2))
    return K

def mmd_distance(mu1, sigma1, mu2, sigma2, num_samples=10000, sigma=1.0):
    """
    计算两个高斯分布之间的MMD距离，基于大数定律的思想
    mu1: shape (n_features,)
    sigma1: shape (n_features,)
    mu2: shape (n_features,)
    sigma2: shape (n_features,)
    num_samples: 抽样点的个数
    sigma: 核宽度参数
    """
    assert mu1.shape == sigma1.shape and mu2.shape == sigma2.shape
    n_features = mu1.shape[0]

    samples1 = np.random.normal(mu1, sigma1, size=(num_samples, n_features))
    samples2 = np.random.normal(mu2, sigma2, size=(num_samples, n_features))

    K_xx = gaussian_kernel(samples1, samples1, sigma=sigma)
    K_yy = gaussian_kernel(samples2, samples2, sigma=sigma)
    K_xy = gaussian_kernel(samples1, samples2, sigma=sigma)
    mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    return mmd

# loading your mu and log_var
test = torch.load("your gaussian distribution 1" ,map_location=torch.device('cpu'))
train_ori = torch.load("your gaussian distribution 2" ,map_location=torch.device('cpu'))


mu1 = test["mu"].detach().numpy()
log_var1 = test["log_var"].detach().numpy()

mu2 = train_ori["mu"].detach().numpy()
log_var2 = train_ori["log_var"].detach().numpy()


# Assuming that mu and log_var are two 2D arrays, with each row representing the distribution parameters of the latent variables for a sample. 

# Calculate the mean and standard deviation of the latent variable distribution for the entire dataset.

mu1 = np.mean(mu1, axis=0)

log_var1 = np.mean(log_var1, axis=0)

# Calculate the standard deviation.
sigma1 = np.sqrt(np.exp(log_var1))

mu2 = np.mean(mu2, axis=0)

log_var2 = np.mean(log_var2, axis=0)

# Calculate the standard deviation.
sigma2 = np.sqrt(np.exp(log_var2))

js = gaussian_js(mu1, sigma1, mu2, sigma2)

mmd = mmd_distance(mu1, sigma1, mu2, sigma2)


print(js)
print(mmd)
