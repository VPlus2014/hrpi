function r = quat_mult(p, q)
    p0 = p(1); % 标量部分
    p_vec = p(2:4); % 向量部分
    
    q0 = q(1); % 标量部分
    q_vec = q(2:4); % 向量部分

    % 计算四元数乘法
    r0 = p0 * q0 - dot(p_vec, q_vec); % 标量部分
    r_vec = p0 * q_vec + q0 * p_vec + cross(p_vec, q_vec); % 向量部分
    
    % 输出结果四元数
    r = [r0; r_vec];
end
