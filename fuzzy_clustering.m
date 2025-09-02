function [U, V, J_history] = fuzzy_clustering(X, S, K, lambda, max_iter, tol)
% 高效模糊聚类算法实现
% 输入:
%   X - 数据矩阵 (T x D)，T是样本数，D是特征维度
%   S - 预计算的相似度矩阵 (T x T)
%   K - 聚类数目
%   lambda - 正则化参数
%   max_iter - 最大迭代次数
%   tol - 收敛阈值
% 输出:
%   U - 隶属度矩阵 (T x K)
%   V - 聚类中心 (K x D)
%   J_history - 目标函数值历史

[T, D] = size(X);

% 初始化隶属度矩阵U
U = rand(T, K);
U = U ./ sum(U, 2); % 归一化

% 预计算一些常量
S_sum = sum(S, 2); % 每行的和，用于简化计算
S_diag = diag(S);  % 对角线元素

% 初始化目标函数历史
J_history = zeros(max_iter, 1);

% 预分配内存
distances = zeros(T, K);
U_prev = U;

for iter = 1:max_iter
    % 1. 更新聚类中心V
    V = (U.^2)' * X ./ sum(U.^2, 1)';
    
    % 2. 计算距离矩阵 (优化计算，避免循环)
    distances = pdist2(X, V, 'squaredeuclidean');
    
    % 3. 更新隶属度矩阵U (使用矩阵运算优化)
    % 预计算公共项
    common_term = lambda * (bsxfun(@times, S_sum, U) - S*U);
    
    % 计算非归一化隶属度
    U_star = 1 ./ (distances + 2*lambda*bsxfun(@minus, S_sum.*U, S*U));
    
    % 处理可能的零距离情况
    U_star(isinf(U_star)) = 1e10;
    U_star(isnan(U_star)) = 0;
    
    % 归一化隶属度
    U = U_star ./ sum(U_star, 2);
    
    % 4. 计算目标函数值
    term1 = sum(sum(U.^2 .* distances));
    term2 = lambda * sum(sum(S .* pdist2(U, U, 'squaredeuclidean')));
    J = term1 + term2;
    J_history(iter) = J;
    
    % 5. 检查收敛条件
    if iter > 1 && norm(U - U_prev, 'fro') < tol
        break;
    end
    U_prev = U;
end

% 截断未使用的历史记录
J_history = J_history(1:iter);

end