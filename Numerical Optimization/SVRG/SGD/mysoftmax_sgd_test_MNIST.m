clear
clc
close all
load MNISTdata % ����MNIST���ݼ�
% ׼ȷ��Ϊ��ȷ������/��������
label = labels(1:9000); % ѵ������ǩ
X = [ones(length(label),1),[inputData(:,1:9000)]']; % ѵ�����������
label_test = labels(9001:end); % ���Լ���ǩ
X_test = [ones(length(label_test),1),[inputData(:,9001:end)]']; % ���Լ��������

lambda = 0.001; % Ȩ��˥������Weight decay parameter
alpha = 0.05; % ѧϰ��
MAX_ITR=50; % ����������

[theta,test_pre,rate,cost] = mysoftmax_sgd(X_test,X,label,lambda,alpha,MAX_ITR);

index_t = find(label_test==test_pre); % �ҳ�Ԥ����ȷ��������λ��
rate_test = length(index_t)/length(label_test); % ���Լ�׼ȷ��
