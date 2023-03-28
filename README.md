# Universal-KL-JS-divergence
An universal implementation for KL/JS divergence calculation even if the dimensions between source domain and target domain are inconsistent.

Also apply an universal implementation for mmd distance calculation even if the dimensions between source domain and target domain are inconsistent.

发现很少有人关注域泛化或域迁移任务下样本数量或维度不一致的两个域之间的KL，JS散度计算问题。
抛砖引玉给出一种基于权重重分配和贝叶斯公式计算源域目标域边缘分布的KL,JS散度计算方法，可用于大部分场景。
