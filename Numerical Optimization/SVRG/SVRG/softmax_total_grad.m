function thetagrad = softmax_total_grad(m,X,label_extend,P,lambda,theta)
% �����ܵ��ݶȾ���
% X��m*p�������mΪ����������pΪ���ϳ�����֮������Ը���
% lambda��Ȩ��˥������weight decay parameter
% theta��p*kϵ������kΪ��ǩ�����
% thetagrad���ݶȾ���
% P��m*k������ʾ���P��i��j����ʾ��i���������б�Ϊ��j��ĸ���
% label_extend:��ǩʾ�Ծ���
thetagrad = -1/m*X'*(label_extend-P)+lambda*theta;