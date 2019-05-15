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
uf=10; % ���ε����еĸ���Ƶ��

[theta7,test_pre7,rate7,cost7] = mysoftmax_svrg1(uf,X_test,X,label,lambda,alpha,MAX_ITR);

index_t7 = find(label_test==test_pre7); % �ҳ�Ԥ����ȷ��������λ��
rate_test7 = length(index_t7)/length(label_test); % ���Լ�׼ȷ��