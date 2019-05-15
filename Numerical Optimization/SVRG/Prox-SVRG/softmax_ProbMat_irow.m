function P = softmax_ProbMat_irow(i,X,m,label_extend,theta)
% 用于计算分类概率矩阵第i行
% X：m*p输入矩阵，m为案例个数，p为加上常数项之后的属性个数
% label：m*1标签向量（数值型）
% theta：p*k系数矩阵，k为标签类别数
% 计算预测概率矩阵
P = zeros(m,size(label_extend,2));
P(i,:) = exp(X(i,:)*theta)/sum(exp(X(i,:)*theta));
