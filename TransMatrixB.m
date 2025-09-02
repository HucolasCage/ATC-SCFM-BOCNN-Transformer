A=ATC(1:N-1,1:N-1).*dCor;
A_norm=zeros((N-1),(N-1));

for i=1:N-1
A_norm(:,i)=minmax_normalize(A(:,i));
end

% 计算每列的和
col_sums = sum(A_norm, 1);

% 处理零和列 - 将它们归一化为均匀分布
zero_cols = (col_sums == 0);
B = A_norm;
B(:, ~zero_cols) = A_norm(:, ~zero_cols) ./ col_sums(~zero_cols);
B(:, zero_cols) = 1/size(A_norm, 1); % 零和列赋值为均匀值


%计算概率
R = rank_power_method(B, R0, 10^(-7));

%排序
[y,z]=sort(R,'descend');

%制作特征排序后的数据集
[L0,W]=size(Dataa);
data_FR=zeros(L0,W);
data_FR(:,end)=Dataa(:,end);
for i=1:W-1
data_FR(:,i)=Data_feature(:,z(i));
end

