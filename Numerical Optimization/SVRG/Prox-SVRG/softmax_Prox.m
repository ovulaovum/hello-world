function thetak = softmax_Prox(thetak_1,yita_k,beta1,epsil,lambda,vk)
% ����������Prox������ֵ�����α�дΪ�˼�㣬�������ݶ��½�����
% thetak_1����һ�������Ľ��
% yita_k��ѧϰ�ʣ���������Ϊ�˼��������óɳ�����
% eps���ݶȵľ���
% beta1��������õ�GD�Ĳ���
thetak = thetak_1;
p = yita_k*vk+lambda*thetak; %��ʼ���ݶ�����
while(norm(p)>epsil)
    thetak=thetak-beta1*p;
    p = thetak-thetak_1+yita_k*vk+lambda*thetak;
end