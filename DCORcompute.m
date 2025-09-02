

for i=1:N-1
    Data_norm(:,i)=minmax_normalize(Data(:,i));
end

%计算DCOR矩阵
for i=1:N-1
    for j=i+1:N-1
        dCor(i,j) = distcorr_optimized(Data_norm(:,i),Data_norm(:,j));
        dCor(j,i) = dCor(i,j);
        dCor(i,i) = 1;
    end
end

