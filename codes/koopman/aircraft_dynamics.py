import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class RigidBodyAircraft:
    """
    刚体飞机动力学模型
    状态向量: [u, v, w, p, q, r, phi, theta, psi, x, y, z]
    u,v,w: 机体坐标系下的线速度
    p,q,r: 机体坐标系下的角速度
    phi,theta,psi: 欧拉角 (roll, pitch, yaw)
    x,y,z: 地面坐标系下的位置
    """
    
    def __init__(self, mass=1000.0, inertia_matrix=None):
        self.mass = mass  # 飞机质量 (kg)
        
        # 惯性矩阵 (kg*m^2)
        if inertia_matrix is None:
            self.I = np.array([
                [1000.0, 0.0, 0.0],      # Ixx
                [0.0, 3000.0, 0.0],      # Iyy  
                [0.0, 0.0, 3500.0]       # Izz
            ])
        else:
            self.I = inertia_matrix
        
        self.I_inv = np.linalg.inv(self.I)
        
        # 重力加速度
        self.g = 9.81
        
        # 气动参数
        self.rho = 1.225  # 空气密度 (kg/m^3)
        self.S = 16.0     # 参考面积 (m^2)
        self.c = 1.5      # 平均弦长 (m)
        self.b = 10.0     # 翼展 (m)
        
        # 气动系数
        self.CL0 = 0.1    # 升力系数常数项
        self.CLa = 5.0    # 升力系数攻角导数
        self.CD0 = 0.02   # 阻力系数常数项
        self.CDa = 0.1    # 阻力系数攻角导数
        
    def dynamics(self, state, t, controls):
        """
        飞机动力学方程
        state: [u, v, w, p, q, r, phi, theta, psi, x, y, z]
        controls: [delta_e, delta_a, delta_r, thrust]
        """
        # 提取状态变量
        u, v, w = state[0:3]        # 机体坐标系线速度
        p, q, r = state[3:6]        # 机体坐标系角速度  
        phi, theta, psi = state[6:9] # 欧拉角
        x, y, z = state[9:12]       # 地面坐标系位置
        
        # 控制输入
        delta_e, delta_a, delta_r, thrust = controls
        
        # 计算空速和攻角
        V = np.sqrt(u**2 + v**2 + w**2)
        if V > 0.1:
            alpha = np.arctan2(w, u)  # 攻角
            beta = np.arcsin(v/V) if abs(v/V) <= 1 else 0  # 侧滑角
        else:
            alpha = beta = 0
            
        # 动压
        q_dyn = 0.5 * self.rho * V**2
        
        # 气动力计算 (机体坐标系)
        CL = self.CL0 + self.CLa * alpha + 0.5 * delta_e
        CD = self.CD0 + self.CDa * abs(alpha)
        
        # 升力和阻力 (风轴)
        L = q_dyn * self.S * CL
        D = q_dyn * self.S * CD
        
        # 转换到机体坐标系
        Fx = thrust - D * np.cos(alpha) - L * np.sin(alpha)
        Fy = q_dyn * self.S * 0.1 * beta + 0.2 * delta_r  # 侧向力
        Fz = -L * np.cos(alpha) + D * np.sin(alpha)
        
        # 重力在机体坐标系的分量
        Fx += -self.mass * self.g * np.sin(theta)
        Fy += self.mass * self.g * np.cos(theta) * np.sin(phi)
        Fz += self.mass * self.g * np.cos(theta) * np.cos(phi)
        
        # 气动力矩 (机体坐标系)
        Mx = q_dyn * self.S * self.b * (0.05 * beta + 0.1 * delta_a)  # 滚转力矩
        My = q_dyn * self.S * self.c * (0.1 * alpha + 0.2 * delta_e)  # 俯仰力矩
        Mz = q_dyn * self.S * self.b * (-0.05 * beta + 0.1 * delta_r) # 偏航力矩
        
        # 力的方程 (机体坐标系)
        u_dot = Fx/self.mass - q*w + r*v
        v_dot = Fy/self.mass - r*u + p*w  
        w_dot = Fz/self.mass - p*v + q*u
        
        # 力矩方程
        omega = np.array([p, q, r])
        M = np.array([Mx, My, Mz])
        omega_dot = self.I_inv @ (M - np.cross(omega, self.I @ omega))
        p_dot, q_dot, r_dot = omega_dot
        
        # 运动学方程 (欧拉角变化率)
        phi_dot = p + (q*np.sin(phi) + r*np.cos(phi)) * np.tan(theta)
        theta_dot = q*np.cos(phi) - r*np.sin(phi)
        psi_dot = (q*np.sin(phi) + r*np.cos(phi)) / np.cos(theta) if abs(np.cos(theta)) > 0.01 else 0
        
        # 位置方程 (地面坐标系)
        # 机体到地面坐标系转换矩阵
        R = self.rotation_matrix(phi, theta, psi)
        vel_body = np.array([u, v, w])
        vel_earth = R @ vel_body
        
        x_dot, y_dot, z_dot = vel_earth
        
        # 返回状态导数
        state_dot = np.array([
            u_dot, v_dot, w_dot,      # 线速度导数
            p_dot, q_dot, r_dot,      # 角速度导数  
            phi_dot, theta_dot, psi_dot, # 欧拉角导数
            x_dot, y_dot, z_dot       # 位置导数
        ])
        
        return state_dot
    
    def rotation_matrix(self, phi, theta, psi):
        """机体到地面坐标系的旋转矩阵"""
        cphi, sphi = np.cos(phi), np.sin(phi)
        ctheta, stheta = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        
        R = np.array([
            [ctheta*cpsi, sphi*stheta*cpsi - cphi*spsi, cphi*stheta*cpsi + sphi*spsi],
            [ctheta*spsi, sphi*stheta*spsi + cphi*cpsi, cphi*stheta*spsi - sphi*cpsi],
            [-stheta, sphi*ctheta, cphi*ctheta]
        ])
        
        return R
    
    def simulate(self, initial_state, time_span, controls_func, dt=0.01):
        """
        仿真飞机运动
        initial_state: 初始状态 [u,v,w,p,q,r,phi,theta,psi,x,y,z]
        time_span: 时间范围 (start, end)
        controls_func: 控制输入函数 controls = f(t)
        """
        t = np.arange(time_span[0], time_span[1], dt)
        
        def dynamics_wrapper(state, t):
            controls = controls_func(t)
            return self.dynamics(state, t, controls)
        
        # 数值积分
        trajectory = odeint(dynamics_wrapper, initial_state, t)
        
        return t, trajectory
    
    def trim_condition(self, V_trim=50.0, gamma=0.0):
        """
        计算配平条件
        V_trim: 配平空速 (m/s)
        gamma: 航迹角 (rad)
        """
        # 简化的配平计算
        alpha_trim = (self.mass * self.g) / (0.5 * self.rho * V_trim**2 * self.S * self.CLa)
        theta_trim = alpha_trim + gamma
        
        # 配平状态
        trim_state = np.zeros(12)
        trim_state[0] = V_trim * np.cos(alpha_trim)  # u
        trim_state[2] = V_trim * np.sin(alpha_trim)  # w
        trim_state[7] = theta_trim                   # theta
        
        # 配平控制
        delta_e_trim = -0.1 * alpha_trim  # 简化升降舵配平
        thrust_trim = 0.5 * self.rho * V_trim**2 * self.S * self.CD0
        
        trim_controls = [delta_e_trim, 0.0, 0.0, thrust_trim]
        
        return trim_state, trim_controls

# 示例控制函数
def step_control(t):
    """阶跃控制输入"""
    if t < 1.0:
        return [0.0, 0.0, 0.0, 1000.0]  # [delta_e, delta_a, delta_r, thrust]
    elif t < 3.0:
        return [0.1, 0.0, 0.0, 1000.0]  # 升降舵阶跃
    elif t < 5.0:
        return [0.0, 0.1, 0.0, 1000.0]  # 副翼阶跃
    else:
        return [0.0, 0.0, 0.1, 1000.0]  # 方向舵阶跃

def sinusoidal_control(t):
    """正弦波控制输入"""
    return [
        0.05 * np.sin(0.5 * t),      # delta_e
        0.03 * np.sin(0.8 * t),      # delta_a  
        0.02 * np.sin(1.2 * t),      # delta_r
        1000.0 + 200 * np.sin(0.3 * t) # thrust
    ]

if __name__ == "__main__":
    # 测试代码
    aircraft = RigidBodyAircraft()
    
    # 获取配平条件
    trim_state, trim_controls = aircraft.trim_condition()
    print("配平状态:", trim_state)
    print("配平控制:", trim_controls)
    
    # 仿真
    t, trajectory = aircraft.simulate(
        initial_state=trim_state,
        time_span=(0, 10),
        controls_func=sinusoidal_control
    )
    
    # 简单绘图
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2,3,1)
    plt.plot(t, trajectory[:,0])
    plt.title('u (m/s)')
    plt.grid(True)
    
    plt.subplot(2,3,2) 
    plt.plot(t, trajectory[:,1])
    plt.title('v (m/s)')
    plt.grid(True)
    
    plt.subplot(2,3,3)
    plt.plot(t, trajectory[:,2]) 
    plt.title('w (m/s)')
    plt.grid(True)
    
    plt.subplot(2,3,4)
    plt.plot(t, trajectory[:,6]*180/np.pi)
    plt.title('φ (deg)')
    plt.grid(True)
    
    plt.subplot(2,3,5)
    plt.plot(t, trajectory[:,7]*180/np.pi)
    plt.title('θ (deg)')
    plt.grid(True)
    
    plt.subplot(2,3,6)
    plt.plot(t, trajectory[:,8]*180/np.pi)
    plt.title('ψ (deg)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()