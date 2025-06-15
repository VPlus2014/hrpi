# 概述

## 定点导航控制任务

$$
\begin{aligned}
\min_{u\in U^\infty} &&J(x,u;p_d)&\coloneqq\sum_{k=0}^{\infty} l(x_k';p_d)\prod_{i=0}^k m(x_i';p_d)\\
\text{s.t.} && x_0' &= x\\
&& x_{k+1}' &= F(x_k', u_k, \delta t)\\
&& u(k) &\in U(x_k')\\
\end{aligned}
$$

其中

$$
\begin{aligned}
m(x;p_d) &\coloneqq \mathbb{I}\{\|p(x)-p_d\|_2>\epsilon\}, \epsilon>0\\
l(x,p_d) &\coloneqq \frac{1}{2}\|p(x)-p_d\|_2^2\\
\end{aligned}
$$

递推关系

$$
\begin{aligned}
J(x,u;p_d)&=m(x;p_d)\left[l(x;p_d)+J(F(x,u_0,\delta t),S u;p_d)\right]\\
\end{aligned}
$$

Bellman方程

$$
\begin{aligned}
J_*(x;p_d)&\coloneqq \min_{u\in U^\infty} J(x,u;p_d)\\
J_*(x;p_d)&= m(x;p_d)\left[l(x;p_d)+\min_{u_0\in U(x)} J_*(F(x,u_0,\delta t);p_d)\right]\\
\end{aligned}
$$
