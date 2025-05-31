function vect_prime = quat_rota(quat, vect)
    % quat 是四元数 [q0, q1, q2, q3]
    % vect 是向量 [v_x, v_y, v_z]
    
    q0 = quat(1);  % 四元数的标量部分
    q_vec = quat(2:4);  % 四元数的向量部分
    
    % 将输入向量转换为四元数（纯四元数，实部为零）
    v_quat = [0; vect];  % 四元数表示的向量 [0, v_x, v_y, v_z]
    
    % 计算四元数的共轭 (共轭是改变向量部分的符号)
    q_conj = [q0; -q_vec];
    
    % 四元数旋转：q * v * q^{-1}
    temp = quat_mult(quat, v_quat);  % 四元数乘法: Q * V
    v_rot_quat = quat_mult(temp, q_conj);  % 四元数乘法: (Q * V) * Q^{-1}
    
    % 结果是一个四元数，取其向量部分作为旋转后的向量
    vect_prime = v_rot_quat(2:4);  % 提取旋转后的向量部分
end

