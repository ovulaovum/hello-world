function cost = softmax_cost(m,label_extend,P,lambda,theta)
% 用于计算代价函数
% m为案例个数
% label_extend:标签示性矩阵
% lambda：权重衰减参数weight decay parameter
% theta：p*k系数矩阵，k为标签类别数
% cost：总代价函数值
% P：m*k分类概率矩阵，P（i，j）表示第i个样本被判别为第j类的概率
cost = -1/m*[label_extend(:)]'*log(P(:))+lambda/2*sum(theta(:).^2);