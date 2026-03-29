function quat_prime = quat_conj(quat)   
    q0 = quat(1);  % 四元数的标量部分
    q_vec = quat(2:4);  % 四元数的向量部分
    
    % 计算四元数的共轭 (共轭是改变向量部分的符号)
    quat_prime = [q0; -q_vec];
end

