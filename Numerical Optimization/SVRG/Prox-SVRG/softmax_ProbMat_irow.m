function P = softmax_ProbMat_irow(i,X,m,label_extend,theta)
% ���ڼ��������ʾ����i��
% X��m*p�������mΪ����������pΪ���ϳ�����֮������Ը���
% label��m*1��ǩ��������ֵ�ͣ�
% theta��p*kϵ������kΪ��ǩ�����
% ����Ԥ����ʾ���
P = zeros(m,size(label_extend,2));
P(i,:) = exp(X(i,:)*theta)/sum(exp(X(i,:)*theta));
