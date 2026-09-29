"""MicroDuck 仿真环境与底层控制器，原样取自 homework1。

这份代码在 HW2 里**不训练也不修改**：它提供 Genesis 环境和 PPO actor 的网络定义，
后者用于加载 HW1 训好的 checkpoint 作为冻结的底层控制器。

只保留了 HW2 需要的部分——`algorithm` 下的 PPO、rollout storage、GAE、logger
都属于训练底层策略的代码，未收录。
"""
