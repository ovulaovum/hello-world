function P = softmax_ProbMat(X,m,label_extend,theta)
%用于计算分类概率矩阵
% X：m*p输入矩阵，m为案例个数，p为加上常数项之后的属性个数
% label：m*1标签向量（数值型）
% theta：p*k系数矩阵，k为标签类别数
% 计算预测概率矩阵
P = zeros(m,size(label_extend,2));
for smp = 1:m
    P(smp,:) = exp(X(smp,:)*theta)/sum(exp(X(smp,:)*theta));
end