function cost = softmax_cost(m,label_extend,P,lambda,theta)
% ���ڼ�����ۺ���
% mΪ��������
% label_extend:��ǩʾ�Ծ���
% lambda��Ȩ��˥������weight decay parameter
% theta��p*kϵ������kΪ��ǩ�����
% cost���ܴ��ۺ���ֵ
% P��m*k������ʾ���P��i��j����ʾ��i���������б�Ϊ��j��ĸ���
cost = -1/m*[label_extend(:)]'*log(P(:))+lambda/2*sum(theta(:).^2);