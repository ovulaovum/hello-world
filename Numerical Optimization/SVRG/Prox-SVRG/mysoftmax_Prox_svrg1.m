function [theta,test_pre,rate,cost] = mysoftmax_Prox_svrg1(uf,beta1,epsil,X_test,X,label,lambda,alpha,MAX_ITR,varargin)
% 本代码编写参考了：https://blog.csdn.net/u013337691/article/details/51784436。
% �ú�������svrg�㷨ʵ��softmax�ع�
% ���÷�ʽ��[theta,test_pre,rate] = mysoftmax_gd(X_test,X,label,lambda,alpha,MAX_ITR,varargin)
% X_test��������������
% X��ѵ���������ݣ���֯Ϊm*p����mΪ����������pΪ���ϳ�����֮������Ը���
% label��ѵ�����ݱ�ǩ����֯Ϊm*1��������ֵ�ͣ�
% lambda��Ȩ��˥������weight decay parameter
% alpha���ݶ��½�ѧϰ����
% uf��update frequency
% MAX_ITR������������
% varargin����ѡ�����������ʼ������thetaϵ�����������룬��Ĭ�����ѡȡ
% theta���ݶ��½�����thetaϵ��Ѱ�Ž��
% test_pre����������Ԥ���ǩ
% rate��ѵ�����ݻ�����ȷ��
%% svrg�㷨Ѱ�����Ž�
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
label_extend = softmax_labext(label);
cost=zeros(MAX_ITR*(uf+1),1); % ����׷�ٴ��ۺ�����ֵ
i=0;
for k=1:MAX_ITR
    i=i+1;
    P = softmax_ProbMat(X,m,label_extend,theta);
    v=softmax_total_grad(m,X,label_extend,P);
    cost(i)= softmax_cost(m,label_extend,P,lambda,theta);
    omega = theta;
    for t=1:uf
        i=i+1;
        r=randperm(m,1);%random pick munber
        vk = softmax_vk(X,r,label_extend,omega,theta,v,m);
        omega = softmax_Prox(omega,alpha,beta1,epsil,lambda,vk);
        P = softmax_ProbMat(X,m,label_extend,omega);
        cost(i)= softmax_cost(m,label_extend,P,lambda,omega);
    end
    theta = omega;
end
%% ����Ԥ��
Probit = softmax_ProbMat(X,m,label_extend,theta);
[~,label_pre] = max(Probit,[],2);
index = find(label==label_pre); % �ҳ�Ԥ����ȷ��������λ��
rate = length(index)/m; % ����Ԥ�⾫��
%% ���ƴ��ۺ���ͼ
figure('Name','���ۺ���ֵ�仯ͼ');
plot(0:i-1,cost)
xlabel('�ݶȸ��´���'); ylabel('���ۺ���ֵ')
title('���ۺ���ֵ�仯ͼ');% ���ƴ��ۺ���ֵ�仯ͼ
%% ��������Ԥ��
[mt,pt] = size(X_test);
Probit_t = zeros(mt,length(unique(label)));
for smpt = 1:mt
    Probit_t(smpt,:) = exp(X_test(smpt,:)*theta)/sum(exp(X_test(smpt,:)*theta));
end
[~,test_pre] = max(Probit_t,[],2);
