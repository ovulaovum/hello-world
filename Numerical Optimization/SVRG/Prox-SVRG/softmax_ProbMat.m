function P = softmax_ProbMat(X,m,label_extend,theta)
%���ڼ��������ʾ���
% X��m*p�������mΪ����������pΪ���ϳ�����֮������Ը���
% label��m*1��ǩ��������ֵ�ͣ�
% theta��p*kϵ������kΪ��ǩ�����
% ����Ԥ����ʾ���
P = zeros(m,size(label_extend,2));
for smp = 1:m
    P(smp,:) = exp(X(smp,:)*theta)/sum(exp(X(smp,:)*theta));
end