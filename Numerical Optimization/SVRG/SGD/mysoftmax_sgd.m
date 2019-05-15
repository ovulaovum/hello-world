function [theta,test_pre,rate,cost] = mysoftmax_sgd(X_test,X,label,lambda,alpha,MAX_ITR,varargin)
% 参考https://blog.csdn.net/u013337691/article/details/51784436
% 该函数用于实现sgd的softmax函数
% 调用方式：[theta,test_pre,rate] = mysoftmax_gd(X_test,X,label,lambda,alpha,MAX_ITR,varargin)
% X_test：测试输入数据
% X：训练输入数据，组织为m*p矩阵，m为案例个数，p为加上常数项之后的属性个数
% label：训练数据标签，组织为m*1向量（数值型）
% lambda权重衰减参数weight decay parameter
% alpha：学习率
% MAX_ITR：最大迭代次数
% varargin：可选参数，输入初始迭代的theta系数，若不输入，则默认随机选取
% theta：梯度下降法的theta系数寻优结果
% test_pre：测试数据预测标签
% rate：训练数据回判准确率
%% sgd寻优
Nin=length(varargin);
if Nin>1
    error('输入太多参数') % 若可选输入参数超过1个，则报错
end
[m,p] = size(X);
numClasses = length(unique(label)); % 求取标签类别数
if Nin==0
    theta = 0.005*randn(p,numClasses); % 若没有输入可选参数，则随机初始化系数
else
    theta=varargin{1}; % 若有输入可选参数，则将其设定为初始theta系数
end
label_extend = softmax_labext(label); % 扩展标签向量
cost=zeros(MAX_ITR,1); % 用于追踪代价函数的值
for k=1:MAX_ITR
    P = softmax_ProbMat(X,m,label_extend,theta); % 计算概率矩阵
    cost(k)= softmax_cost(m,label_extend,P,lambda,theta); % 记录代价函数值
    r=randperm(m,1);%random pick munber
    theta = theta-alpha*softmax_comp_grad(r,X,label_extend,P,lambda,theta); % 更新系数
end
%%  回判预测
Probit = softmax_ProbMat(X,m,label_extend,theta);
[~,label_pre] = max(Probit,[],2);
index = find(label==label_pre); % 找出预测正确样本的位置
rate = length(index)/m; % 计算预测准确率
%% 绘制代价函数图
figure('Name','代价函数值变化图');
plot(0:MAX_ITR-1,cost)
xlabel('迭代次数'); ylabel('代价函数值')
title('代价函数值变化图');% 绘制代价函数值变化图
%% 测试数据预测
[mt,~] = size(X_test);
Probit_t = zeros(mt,length(unique(label)));
for smpt = 1:mt
    Probit_t(smpt,:) = exp(X_test(smpt,:)*theta)/sum(exp(X_test(smpt,:)*theta));
end
[~,test_pre] = max(Probit_t,[],2);
