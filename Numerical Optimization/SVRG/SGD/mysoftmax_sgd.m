function [theta,test_pre,rate,cost] = mysoftmax_sgd(X_test,X,label,lambda,alpha,MAX_ITR,varargin)
% �ο�https://blog.csdn.net/u013337691/article/details/51784436
% �ú�������ʵ��sgd��softmax����
% ���÷�ʽ��[theta,test_pre,rate] = mysoftmax_gd(X_test,X,label,lambda,alpha,MAX_ITR,varargin)
% X_test��������������
% X��ѵ���������ݣ���֯Ϊm*p����mΪ����������pΪ���ϳ�����֮������Ը���
% label��ѵ�����ݱ�ǩ����֯Ϊm*1��������ֵ�ͣ�
% lambdaȨ��˥������weight decay parameter
% alpha��ѧϰ��
% MAX_ITR������������
% varargin����ѡ�����������ʼ������thetaϵ�����������룬��Ĭ�����ѡȡ
% theta���ݶ��½�����thetaϵ��Ѱ�Ž��
% test_pre����������Ԥ���ǩ
% rate��ѵ�����ݻ���׼ȷ��
%% sgdѰ��
Nin=length(varargin);
if Nin>1
    error('����̫�����') % ����ѡ�����������1�����򱨴�
end
[m,p] = size(X);
numClasses = length(unique(label)); % ��ȡ��ǩ�����
if Nin==0
    theta = 0.005*randn(p,numClasses); % ��û�������ѡ�������������ʼ��ϵ��
else
    theta=varargin{1}; % ���������ѡ�����������趨Ϊ��ʼthetaϵ��
end
label_extend = softmax_labext(label); % ��չ��ǩ����
cost=zeros(MAX_ITR,1); % ����׷�ٴ��ۺ�����ֵ
for k=1:MAX_ITR
    P = softmax_ProbMat(X,m,label_extend,theta); % ������ʾ���
    cost(k)= softmax_cost(m,label_extend,P,lambda,theta); % ��¼���ۺ���ֵ
    r=randperm(m,1);%random pick munber
    theta = theta-alpha*softmax_comp_grad(r,X,label_extend,P,lambda,theta); % ����ϵ��
end
%%  ����Ԥ��
Probit = softmax_ProbMat(X,m,label_extend,theta);
[~,label_pre] = max(Probit,[],2);
index = find(label==label_pre); % �ҳ�Ԥ����ȷ������λ��
rate = length(index)/m; % ����Ԥ��׼ȷ��
%% ���ƴ��ۺ���ͼ
figure('Name','���ۺ���ֵ�仯ͼ');
plot(0:MAX_ITR-1,cost)
xlabel('��������'); ylabel('���ۺ���ֵ')
title('���ۺ���ֵ�仯ͼ');% ���ƴ��ۺ���ֵ�仯ͼ
%% ��������Ԥ��
[mt,~] = size(X_test);
Probit_t = zeros(mt,length(unique(label)));
for smpt = 1:mt
    Probit_t(smpt,:) = exp(X_test(smpt,:)*theta)/sum(exp(X_test(smpt,:)*theta));
end
[~,test_pre] = max(Probit_t,[],2);
