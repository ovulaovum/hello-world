function thetagrad = softmax_total_grad(m,X,label_extend,P,lambda,theta)
% 计算总的梯度矩阵
% X：m*p输入矩阵，m为案例个数，p为加上常数项之后的属性个数
% lambda：权重衰减参数weight decay parameter
% theta：p*k系数矩阵，k为标签类别数
% thetagrad：梯度矩阵
% P：m*k分类概率矩阵，P（i，j）表示第i个样本被判别为第j类的概率
% label_extend:标签示性矩阵
thetagrad = -1/m*X'*(label_extend-P)+lambda*theta;