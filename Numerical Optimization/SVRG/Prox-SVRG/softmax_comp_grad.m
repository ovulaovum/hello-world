function stochgrad = softmax_comp_grad(r,X,label_extend,P)
%X'(~,r)*(LE-P)(r,~)����������ݶȵĵ�r����ͷ���
% ���ڼ���һ�������ݶ� component gradient
% X��m*p�������mΪ����������pΪ���ϳ�����֮������Ը���
% label_extend����ǩʾ�Ծ���
% P��������ʾ���
% lambda��Ȩ��˥������weight decay parameter
% theta��p*kϵ������kΪ��ǩ�����
% rΪҪ��ķ������±�
Dif = label_extend-P;
stochgrad = -X(r,:)'*Dif(r,:);