# 四元数运算的矩阵化

- 共轭

$$
\begin{aligned}
h^* &=J h\\
J &= \begin{bmatrix}
    1 & 0 \\
    0 & -I_3\\
\end{bmatrix} \\
\end{aligned}
$$

- 乘法

$$
\begin{aligned}
Q\otimes h &= M(Q)h \\
M(Q)&=\begin{bmatrix}
    \mathrm{Re}(Q) & -(\mathrm{Im}(Q))^\top \\
    \mathrm{Im}(Q) & \mathrm{Re}(Q)I_3 + (\mathrm{Im}(Q))_\times \\
\end{bmatrix}\\
M(Q^*) &= (M(Q))^\top\\
h\otimes Q &= J (M(Q))^\top J h \\
\end{aligned}
$$

- 旋转

$$
\begin{aligned}
Q\otimes h\otimes Q^* &= (Q\otimes (Q\otimes h)^*)^*\\
&= (J M(Q) (J M(Q) h))\\
&= (J M(Q) J M(Q)) h \\
J M(Q) &= \begin{bmatrix}
    \mathrm{Re}(Q) & -(\mathrm{Im}(Q))^\top \\
    -\mathrm{Im}(Q) & \mathrm{Re}(Q)I_3-(\mathrm{Im}(Q))_\times \\
\end{bmatrix}\\
\end{aligned}
$$
