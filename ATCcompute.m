ATC=zeros(N0,N0);
tao=off_diag_median(PTE);

for i = 1:N
    for j = 1:N
        if i~=j
            ATC(i,j)=PTE(i,j)/((PTE(i,j)+PTE(j,i))*(1+exp(-PTE(i,j)/tao)));%求ATC系数矩阵
        end
    end
end

mu= off_diag_mean(ATC);
delta= off_diag_std(ATC);

for i = 1:N
    for j = 1:N
        if abs(ATC(i,j)-mu) > 6*delta
           
                ATC(i,j)=0;%未经过检验的要变为0

        end
    end
end


R0=ATC(1:N-1,N);
for i=1:N-1
    if R0(i) < 0.25%制作初始向量（即特征与待预测变量间的关系）
        R0(i)=0;%对于特征与待预测变量，如果系数小于0.25，我们认为待预测变量不应该指向特征变量，所以令其为0
    end
end






