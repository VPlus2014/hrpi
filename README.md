# 概述

## 框架

基于 Gymnasium, 追逃场景下的固定翼飞行器组导航控制

工作清单

- 面向训练的sim仿真环境
  - numpy 并行版
    - 导航
    - 追逃
  - cuda 并行版
- sim2real 仿真
  - 未开始
- algorithm
  - MORL(Multi-Objective Reinforcement Learning) 表征学习模块
    - 增量式奖励函数重组&末端权重加载

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

## terminated vs. truncated

$h_1(t,x)=\neg 1\{\text{terminated\;at\;} x\}$
$h_2(t,x)=\neg 1\{\text{truncated\;at\;} t\}$

对于无限视野OCP

$$
V_\pi(k,x)=h_1(k,x)\mathbb{E}\left[(R_k+\gamma V_\pi(k,X_{k+1}))\middle|X_k=x,\pi\right]
$$

$h_2$ 用于控制截断情形下逆时间轴的信息传递。按照仿真逻辑，如果首次出现 truncated, 那么 obs_next 往后的信息将不应该传递到 obs_next 。为方便描述，做出如下约定:

- 每一个时刻状态/观测都有一组对应的 (truncated,terminated) 标志。
- $terminated[t]=1 \implies \forall k>t.(terminated[k]=1)$
- $truncated[t]=1 \implies \forall k>t.(truncated[k]=1 \land terminated[k]=terminated[t])$

```python
vs = critic(obs) * ~terminated # regularized state value, shape=(T+1,dimV)
v1 = vs[:-1] # current state value, shape=(T,dimV)
v2 = vs[1:] # next state value, shape=(T,dimV)
T = v1.shape[-2]
r = r * ~terminated[:-1] # regularized reward for non-terminated states, shape=(T,dimV)
v1_target = r + gamma * v2  # TD(0) estimation

for t in range(T):
    if not truncated[t]: # info propagation
        critic.fit(v1[t], v1_target[t]) # update critic
        actor.fit(obs[t], action[t], v1[t], v1_target[t]) # update policy
    else: # info from v2[t] is invalid, do nothing
        pass
```

### critic 参数更新

critic 参数更新通过 TDE 传递，通过上式得到 TDE(0) 的张量计算形式为

```python
vs = critic(obs) * ~terminated # regularized state value, shape=(T+1,dimV)
v1 = vs[:-1]
v2 = vs[1:]
v1_target = r + gamma * v2
tde_0 = (v1_target-v1) * ~truncated[:-1] # regularized GAE(0) estimation & one-step TD error
tde_0.square().mean().backward() # propagation in TDMSE
```

注意，最终施加在 ```v2[t]``` 上的掩码是 ```(~terminated[t+1]) & (~truncated[t])```

在 GAE/TD($\lambda$) 估计中，信息传递逻辑为

```python
gae = zeros_like(tde_0) # initialize GAE(\lambda) estimation, shape=(T,dimV)
beta = lambda_*gamma # lambda-return factor
horizon = math.inf
for t in range(T):
    if not truncated[t]:
        gae[t] = tde_0[t]
        T_ = int(min(T,t+horizon))
        for k in range(t+1,T_):
            beta_k = beta**(k-t)
            if not truncated[k]:
                gae[t] += beta_k * tde_0[k]
            else:
                break
    else:
        break
v1_target_lambda = v1 + gae # lambda-return estimation
```

等价的向量化计算方法为

```python
beta = lambda_*gamma # lambda-return factor
gae = deepcopy(tde_0) # initialize GAE(\lambda) estimation
use_fwd_form = False # use forward form or backward form
if use_fwd_form: # O(T^2)
    betas = logspace(0,T-1,T,base=beta) # beta**[0,1,...,T-1]
    for t in range(T):
        T_ = int(min(T,t+horizon))
        gae[t] += (tde_0[t+1:T_]*betas[1:T_-t]).sum() # impleted as convolution
else: # O(T)
    for t in reversed(range(T-1)):
        gae[t] += beta * gae[t+1]
v1_target_lambda = v1 + gae

tde_lambda = (v1_target_lambda - v1) * ~truncated[:-1] # regularized TD(\lambda) error estimation
tde_lambda.square().mean().backward() # propagation in TDMSE
```


### actor 参数更新

actor 分为分布型&确定型两种更新方式。

- 对于分布型，一般依赖重要性采样公式

```python
logpa = actor.get_dist(obs)
logpa_old = actor_old.get_dist(obs)
for t in range(T):
    if terminated[t] or truncated[t]:
        pass
    else:
        ratio_t = (logpa[t]-logpa_old[t]).exp() # importance sampling ratio
        actor_loss_t = -(gae[t] * ratio_t).mean() # actor loss
```

等价格式

```python
obs1 = obs[:-1]
logpa = actor.get_dist(obs1, act) # shape=(T,dimA)
logpa_old = actor_old.get_dist(obs1, act)
done = (terminated | truncated)[:-1] # shape=(T,1)
ratio = where(done, 0, (logpa-logpa_old).exp()) # importance sampling ratio
actor_loss = -(ratio*gae).mean() # actor loss
```

### 总结

对于每步仿真的返回信息 ```(obs,act,obs_next,terminated,truncated)```
```terminated``` 是 ```obs_next``` 是否终止， 赋值给 ```buffer_terminated[k+1]```
```truncated``` 是是否采纳 transition after ```obs``` 的信息（包括 ```obs_next```,```action```,```reward``` ），赋值给 ```buffer_truncated[k+1]``` 。

### 余弦退火

$$
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)
$$

###

```((pred)|(target)|(reward))_total```
