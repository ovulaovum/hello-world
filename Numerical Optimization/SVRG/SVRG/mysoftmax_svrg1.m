function [theta,test_pre,rate,cost] = mysoftmax_svrg1(uf,X_test,X,label,lambda,alpha,MAX_ITR,varargin)
% �ο�https://blog.csdn.net/u013337691/article/details/51784436
% �ú�������ʵ��svrg��softmax����
% ���÷�ʽ��[theta,test_pre,rate] = mysoftmax_gd(X_test,X,label,lambda,alpha,MAX_ITR,varargin)
% X_test��������������
% X��ѵ���������ݣ���֯Ϊm*p����mΪ����������pΪ���ϳ�����֮������Ը���
% label��ѵ�����ݱ�ǩ����֯Ϊm*1��������ֵ�ͣ�
% lambdaȨ��˥������weight decay parameter
% alpha��ѧϰ��
% uf�����ε����ĸ���Ƶ��update frequency
% MAX_ITR������������
% varargin����ѡ�����������ʼ������thetaϵ�����������룬��Ĭ�����ѡȡ
% theta���ݶ��½�����thetaϵ��Ѱ�Ž��
% test_pre����������Ԥ���ǩ
% rate��ѵ�����ݻ���׼ȷ��
%% svrgѰ��
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
cost=zeros(MAX_ITR*(uf+1),1); % ����׷�ٴ��ۺ�����ֵ
i=0;
for k=1:MAX_ITR
    i=i+1;
    P = softmax_ProbMat(X,m,label_extend,theta);% ������ʾ���
    mu = softmax_total_grad(m,X,label_extend,P,lambda,theta);% �������������ݶ�
    cost(i)= softmax_cost(m,label_extend,P,lambda,theta);% ��¼���ۺ���ֵ
    omega = theta;
    for t=1:uf
        i=i+1;
        r=randperm(m,1);%random pick munber
        omega = omega-alpha*(softmax_comp_grad(r,X,label_extend,P,lambda,omega)-softmax_comp_grad(r,X,label_extend,P,lambda,theta)+mu);
        P = softmax_ProbMat(X,m,label_extend,omega);
        cost(i)= softmax_cost(m,label_extend,P,lambda,omega);
    end
    theta = omega;
end
%% ����Ԥ��
Probit = softmax_ProbMat(X,m,label_extend,theta);
[~,label_pre] = max(Probit,[],2);% ��ȡ�������ı�ǩ
index = find(label==label_pre); % �ҳ�Ԥ����ȷ������λ��
rate = length(index)/m; % ����Ԥ��׼ȷ��
%% ���ƴ��ۺ���ͼ
figure('Name','���ۺ���ֵ�仯ͼ');
plot(0:i-1,cost) % i = MAX_ITR
xlabel('��������'); ylabel('���ۺ���ֵ')
title('���ۺ���ֵ�仯ͼ');% ���ƴ��ۺ���ֵ�仯ͼ
%% ��������Ԥ��
[mt,~] = size(X_test);
Probit_t = zeros(mt,length(unique(label)));
for smpt = 1:mt
    Probit_t(smpt,:) = exp(X_test(smpt,:)*theta)/sum(exp(X_test(smpt,:)*theta));
end
[~,test_pre] = max(Probit_t,[],2);
