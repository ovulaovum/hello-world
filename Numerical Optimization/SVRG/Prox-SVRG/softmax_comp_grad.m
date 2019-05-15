function stochgrad = softmax_comp_grad(r,X,label_extend,P)
%X'(~,r)*(LE-P)(r,~)就是求和总梯度的第r个求和分量
% 用于计算一个分量梯度 component gradient
% X：m*p输入矩阵，m为案例个数，p为加上常数项之后的属性个数
% label_extend：标签示性矩阵
% P：分类概率矩阵
% lambda：权重衰减参数weight decay parameter
% theta：p*k系数矩阵，k为标签类别数
% r为要求的分量的下标
Dif = label_extend-P;
stochgrad = -X(r,:)'*Dif(r,:);