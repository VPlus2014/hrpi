import torch
from typing import Literal


# 定义基元旋转矩阵
def Lx(angle_rad: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle_rad)
    sin = torch.sin(angle_rad)
    one = torch.ones_like(cos)
    zero = torch.zeros_like(cos)
    
    L_flat = (one, zero, zero, zero, cos, sin, zero, -sin, cos)
    return torch.cat(L_flat, -1).reshape(angle_rad.shape[0:1] + (3, 3))

def Ly(angle_rad: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle_rad)
    sin = torch.sin(angle_rad)
    one = torch.ones_like(cos)
    zero = torch.zeros_like(cos)
    
    L_flat = (cos, zero, -sin, zero, one, zero, sin, zero, cos)
    return torch.cat(L_flat, -1).reshape(angle_rad.shape[0:1] + (3, 3))

def Lz(angle_rad: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle_rad)
    sin = torch.sin(angle_rad)
    one = torch.ones_like(cos)
    zero = torch.zeros_like(cos)
    
    L_flat = (cos, sin, zero, -sin, cos, zero, zero, zero, one)
    return torch.cat(L_flat, -1).reshape(angle_rad.shape[0:1] + (3, 3))

def Qx(angle_rad: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle_rad/2)
    sin = torch.sin(angle_rad/2)
    zero = torch.zeros_like(cos)

    Q_flat = (cos, sin, zero, zero)
    return torch.cat(Q_flat, -1)

def Qy(angle_rad: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle_rad/2)
    sin = torch.sin(angle_rad/2)
    zero = torch.zeros_like(cos)

    Q_flat = (cos, zero, sin, zero)
    return torch.cat(Q_flat, -1)

def Qz(angle_rad: torch.Tensor) -> torch.Tensor:
    """

    Args:
        angle_rad: Input tensor of shape (N, 1).

    Rets:
        
    """
    cos = torch.cos(angle_rad/2)
    sin = torch.sin(angle_rad/2)
    zero = torch.zeros_like(cos)

    Q_flat = (cos, zero, zero, sin)
    return torch.cat(Q_flat, -1)

# @torch.jit.script
def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Rets:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

# @torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Rets:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1)

def mat(q: torch.Tensor) -> torch.Tensor:
    """Computes the matrix of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Rets:
        The matrix of quaternion. Shape is (..., 4, 4).
    """
    q0, q1, q2, q3 = q.unbind(dim=-1)

    mat_flat = (q0, -q1, -q2, -q3, q1, q0, -q3, q2, q2, q3, q0, -q1, q3, -q2, q1, q0)
    return torch.stack(mat_flat, -1).reshape(q0.shape + (4, 4))

def mati(q: torch.Tensor) -> torch.Tensor:
    """Computes the matrix of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Rets:
        The matrix inverse of quaternion. Shape is (..., 4, 4).
    """
    q0, q1, q2, q3 = q.unbind(dim=-1)

    mat_flat = (q0, -q1, -q2, -q3, q1, q0, q3, -q2, q2, -q3, q0, q1, q3, q2, -q1, q0)
    return torch.stack(mat_flat, -1).reshape(q0.shape + (4, 4))

# # @torch.jit.script
def quat_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        p: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Rets:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).
    """
    r = torch.bmm(mat(p), q.unsqueeze(-1)).squeeze(-1)
    return r

def rotaion_matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Get rotation matrix from quaternion.

    Args:
        p: The quaternion in (w, x, y, z). Shape is (..., 4).

    Rets:
        The rotation matrix of quaternion. Shape is (..., 3, 3).
    """
    rotation_matrix = torch.bmm(mati(q), mat(quat_conjugate(q)))

    return rotation_matrix[..., 1:, 1:]

def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion along the last dimension of q and v.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    rotation_matrix = rotaion_matrix_from_quat(q)
    u = torch.bmm(rotation_matrix, v.unsqueeze(-1)).squeeze(-1)

    return u

def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Inverse rotate a vector by a quaternion along the last dimension of q and v.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    rotation_matrix = rotaion_matrix_from_quat(quat_conjugate(q))
    u = torch.bmm(rotation_matrix, v.unsqueeze(-1)).squeeze(-1)

    return u

def quat_enu_ned() -> torch.Tensor:
    return quat_mul(Qz(torch.tensor([[torch.pi/2]])), quat_mul(Qy(torch.tensor([[0]])), Qx(torch.tensor([[torch.pi]]))))

def euler_from_quat(q: torch.Tensor) -> torch.Tensor:
    q0, q1, q2, q3 = q.unbind(dim=-1)

    phi = torch.arctan2(2*(q2*q3+q0*q1), 1-2*(q1*q1+q2*q2))
    theta = torch.arcsin(torch.clamp(-2*(q3*q1-q0*q2), -1, 1))
    psi = torch.arctan2(2*(q1*q2+q0*q3), 1-2*(q2*q2+q3*q3))

    return torch.stack([phi, theta, psi], dim=-1)

def ned2aer(ned: torch.Tensor) -> torch.Tensor:
    n, e, d = torch.unbind(ned, dim=-1)
    az = torch.atan2(e, n)
    r = torch.sqrt(torch.pow(e, 2)+torch.pow(n, 2))
    elev = torch.atan(-d/r)
    slant_range = torch.sqrt(torch.pow(r, 2)+torch.pow(d, 2))

    return torch.stack([az, elev, slant_range], dim=-1)

def aer2ned(aer: torch.Tensor) -> torch.Tensor:
    a, e, r = torch.unbind(aer, dim=-1)
    
    r_prime = r * torch.cos(e)
    n = r_prime * torch.cos(a)
    e = r_prime * torch.sin(a)
    d = -r * torch.sin(e)

    return torch.stack([n, e, d], dim=-1)

def _herp(position_0: torch.Tensor, velocity_0: torch.Tensor, position_1: torch.Tensor, velocity_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    position = torch.cat([position_0, position_1], dim=-1)
    velocity = torch.cat([velocity_0, velocity_1], dim=-1)
    dtype = position.dtype
    device = position.device
    x = torch.cat([
        torch.zeros(size=(position.shape[0], 1), dtype=dtype, device=device),
        torch.ones(size=(position.shape[0], 1), dtype=dtype, device=device),
        ], dim=-1)
    
    r, c1 = x.shape
    _, c2 = t.shape

    x_prime_1 = x.repeat(1, c1).reshape(r, c1, c1).permute(0, 2, 1)
    x_prime_mask = torch.eye(c1, device=device).repeat(r, 1, 1)
    x_prime_2 = x_prime_1 - x_prime_mask * x_prime_1
    x_prime = x_prime_2.unsqueeze(1).repeat(1, c2, 1, 1)
    
    t_prime_1 = t.unsqueeze(2).repeat(1, 1, c1)
    t_prime = t_prime_1.unsqueeze(3).repeat(1, 1, 1, c1)
    
    l_num_1 = t_prime - x_prime
    l_num_mask = torch.eye(c1, device=device).repeat(r * c2, 1, 1).reshape(r, c2, c1, c1)
    l_num_2 = l_num_1 - l_num_mask * l_num_1 + l_num_mask
    l_num = torch.prod(l_num_2, dim=2)

    l_den_1 = torch.prod((x_prime_1.permute(0, 2, 1) - x_prime_1) + x_prime_mask, dim=1)
    l_den = l_den_1.unsqueeze(1).repeat(1, c2, 1)

    l = l_num / l_den

    l_prime_1 = 1.0 / ((x_prime_1.permute(0, 2, 1) - x_prime_1) + x_prime_mask)
    l_prime_2 = torch.sum(l_prime_1, dim=1) - 1
    l_prime = l_prime_2.unsqueeze(1).repeat(1, c2, 1)

    # Create x_prime_3 for functions A and B
    x_prime_3 = x.unsqueeze(1).repeat(1, c2, 1)

    # Calculate function B
    B = (t_prime_1 - x_prime_3) * l * l

    # Calculate function A
    A = (1 - 2 * (t_prime_1 - x_prime_3) * l_prime) * l * l

    # Calculate final result H
    A_prime = position.unsqueeze(1).repeat(1, c2, 1) * A
    B_prime = velocity.unsqueeze(1).repeat(1, c2, 1) * B
    H = torch.sum(A_prime, dim=2) + torch.sum(B_prime, dim=2)

    return H
    
def herp(position_0: torch.Tensor, velocity_0: torch.Tensor, position_1: torch.Tensor, velocity_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    args:
        position_0: shape is [n, 3]
        velocity_0: shape is [n, 3]
        position_1: shape is [n, 3]
        velocity_1: shape is [n, 3]
        t: shape is [n, m]

    rets:
        position: shape is [n, m, 3]
    """
    n = _herp(position_0[..., 0:1], velocity_0[..., 0:1], position_1[..., 0:1], velocity_1[..., 0:1], t)    # n, shape is [n, m]
    e = _herp(position_0[..., 1:2], velocity_0[..., 1:2], position_1[..., 1:2], velocity_1[..., 1:2], t)    # n, shape is [n, m]
    d = _herp(position_0[..., 2:3], velocity_0[..., 2:3], position_1[..., 2:3], velocity_1[..., 2:3], t)    # n, shape is [n, m]

    position = torch.stack([n, e, d], dim=-1)
    return position

def lerp(v_0: torch.Tensor, v_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    v_0 = v_0.unsqueeze(1).repeat(1, t.shape[-1], 1)
    v_1 = v_1.unsqueeze(1).repeat(1, t.shape[-1], 1)
    t = t.unsqueeze(-1).repeat(1, 1, v_0.shape[-1])

    v_t = (1-t)*v_0 + t*v_1
    return v_t

def nlerp(q_0: torch.Tensor, q_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    args:
        q_0: shape is [n, 4]
        q_1: shape is [n, 4]
        t: shape is [n, m]

    rets:
        q_t: shape is [n, m, 4]
    """

    # 钝角检测, 解决双倍覆盖问题
    dot = torch.einsum("ij,ij->i", [q_0, q_1])
    indices = torch.where(dot < 0)[0]
    q_1[indices] *= -1

    q_t = lerp(q_0, q_1, t)
    
    q_t = normalize(q_t)                    # TODO:注意可能会导致问题，有待验证
    return q_t