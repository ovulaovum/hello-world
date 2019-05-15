function [theta,test_pre,rate] = mysoftmax_gd(X_test,X,label,lambda,alpha,MAX_ITR,varargin)
% 本代码编写参考了：https://blog.csdn.net/u013337691/article/details/51784436。
% �ú�������ʵ���ݶ��½���softmax�ع�
% ���÷�ʽ��[theta,test_pre,rate] = mysoftmax_gd(X_test,X,label,lambda,alpha,MAX_ITR,varargin)
% X_test��������������
% X��ѵ���������ݣ���֯Ϊm*p����mΪ����������pΪ���ϳ�����֮������Ը���
% label��ѵ�����ݱ�ǩ����֯Ϊm*1��������ֵ�ͣ�
% lambda��Ȩ��˥������weight decay parameter
% alpha���ݶ��½�ѧϰ����
% MAX_ITR������������
% varargin����ѡ�����������ʼ������thetaϵ�����������룬��Ĭ�����ѡȡ
% theta���ݶ��½�����thetaϵ��Ѱ�Ž��
% test_pre����������Ԥ���ǩ
% rate��ѵ�����ݻ�����ȷ��

% Genlovy Hoo��2016.06.29. genlovhyy@163.com
%% �ݶ��½�Ѱ��
Nin=length(varargin);
if Nin>1
    error('����̫�����') % ����ѡ�����������1�����򱨴�
end
[m,p] = size(X);
numClasses = length(unique(label)); % ��ȡ��ǩ�����
if Nin==0
    theta = 0.005*randn(p,numClasses) % ��û�������ѡ�������������ʼ��ϵ��
else
    theta=varargin{1}; % ���������ѡ�����������趨Ϊ��ʼthetaϵ��
end
cost=zeros(MAX_ITR,1); % ����׷�ٴ��ۺ�����ֵ
for k=1:MAX_ITR
    [cost(k),grad] = softmax_cost_grad(X,label,lambda,theta); % ������ۺ���ֵ���ݶ�
    theta=theta-alpha*grad; % ����ϵ��
end
%% ����Ԥ��
[~,~,Probit] = softmax_cost_grad(X,label,lambda,theta);
[~,label_pre] = max(Probit,[],2);
index = find(label==label_pre); % �ҳ�Ԥ����ȷ��������λ��
rate = length(index)/m; % ����Ԥ�⾫��
%% ���ƴ��ۺ���ͼ
figure('Name','���ۺ���ֵ�仯ͼ');
plot(0:MAX_ITR-1,cost)
xlabel('��������'); ylabel('���ۺ���ֵ')
title('���ۺ���ֵ�仯ͼ');% ���ƴ��ۺ���ֵ�仯ͼ
%% ��������Ԥ��
[mt,pt] = size(X_test);
Probit_t = zeros(mt,length(unique(label)));
for smpt = 1:mt
    Probit_t(smpt,:) = exp(X_test(smpt,:)*theta)/sum(exp(X_test(smpt,:)*theta));
end
[~,test_pre] = max(Probit_t,[],2);
