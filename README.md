# Universal-KL-JS-divergence
An universal implementation for KL/JS divergence calculation even if the dimensions between source domain and target domain are inconsistent.

Also apply an universal implementation for mmd distance calculation even if the dimensions between source domain and target domain are inconsistent.

发现很少有人关注域泛化或域迁移任务下样本数量或维度不一致的两个域之间的KL，JS散度计算问题。
抛砖引玉给出一种基于权重重分配和贝叶斯公式计算源域目标域边缘分布的KL,JS散度计算方法，可用于大部分场景。


Update 2023/4.14

We have uploaded an universal method for calculating Multidimensional Scaling (MMD), Kullback–Leibler (KL), and Jensen–Shannon (JS) divergences in cases where the dimensions are not consistent, taking into account the addition of known Gaussian distribution parameters mu and log_var (which requires conversion and is not equivalent to variance).

增加了已知高斯分布mu和log_var(不是方差，需要转化)参数情况下的维度不一致情况下的通用MMD,KL,JS散度计算方法。
