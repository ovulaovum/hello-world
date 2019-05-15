clear
clc
close all
load MNISTdata % 加载MNIST数据集
% 准确率为正确样本数/总样本数
label = labels(1:9000); % 训练集标签
X = [ones(length(label),1),[inputData(:,1:9000)]']; % 训练集输入矩阵
label_test = labels(9001:end); % 测试集标签
X_test = [ones(length(label_test),1),[inputData(:,9001:end)]']; % 测试集输入矩阵

lambda = 0.001; % 权重衰减参数Weight decay parameter
alpha = 0.05; % 学习率
MAX_ITR=50; % 最大迭代次数

[theta,test_pre,rate,cost] = mysoftmax_sgd(X_test,X,label,lambda,alpha,MAX_ITR);

index_t = find(label_test==test_pre); % 找出预测正确的样本的位置
rate_test = length(index_t)/length(label_test); % 测试集准确率
