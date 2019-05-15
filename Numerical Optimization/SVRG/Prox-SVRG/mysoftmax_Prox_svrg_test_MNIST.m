clear
clc
close all
load MNISTdata % MNIST���ݼ�
% ׼��ѵ��/�������ݼ�
label = labels(1:9000); % ѵ������ǩ
X = [ones(length(label),1),[inputData(:,1:9000)]']; % ѵ������������
label_test = labels(9001:end); % ���Լ���ǩ
X_test = [ones(length(label_test),1),[inputData(:,9001:end)]']; % ������������

lambda = 0.001; % Ȩ��˥������Weight decay parameter
alpha = 0.05; % ѧϰ����
beta1=0.01; %GD��ѧϰ����
epsil=10e-5;
MAX_ITR=50; % ����������
uf=10; %updating frequency

[theta,test_pre,rate,cost] = mysoftmax_Prox_svrg1(uf,beta1,epsil,X_test,X,label,lambda,alpha,MAX_ITR)

index_t = find(label_test==test_pre); % �ҳ�Ԥ����ȷ��������λ��
rate_test = length(index_t)/length(label_test); % ����Ԥ�⾫��
