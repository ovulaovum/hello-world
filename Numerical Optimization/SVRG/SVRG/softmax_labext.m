function label_extend = softmax_labext(label)
% �����ǩʾ�Ծ���
% ��ÿ����ǩ��չΪһ��kά��������kΪ��ǩ���������������i���ڵ�j�࣬��
% label_extend��i��j��= 1������label_extend��i��j��= 0��
label_extend = [full(sparse(label,1:length(label),1))]';