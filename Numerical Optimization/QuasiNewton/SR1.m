function [x,k] = SR1(x,eps,f,g)
%SR1 ��ţ�ٷ�
%��������ʼ������x������eps��Ŀ�꺯��f���ݶ�g
n=length(x);%��ȡά��
gk=g(x);%��ʼ���ݶ�
delta=0.2; %��ʼ�뾶
eta=0.01; %��СԤ���
r=10^(-8);%�ж��������µ�ϵ��
k=0;%��¼��������
B=eye(n,n);%��ʼ��H
% %����һ������ʼ����ɭ���������H
% p=-gk;
% alpha=1;
% while f(x+alpha*p)>fk+10^(-4)*alpha*gk'*p
%     alpha=0.4*alpha;
% end
% s=alpha*p;
% y=g(x+s)-gk;
% H=y'*s/(y'*y)*eye(n,n);

%��ʼ����
while norm(gk)>eps
if k>=10000
    k='maxiterations';%������������ֹ����
    break
end
s=CGSteihaug(eps,delta,gk,B);
y=g(x+s)-gk;
rho=(f(x)-f(x+s))/(-gk'*s-1/2*s'*B*s);
if rho>eta  %����x
    x=x+s;
end
gk=g(x);
if rho>3/4  %���°뾶
    if norm(s)>0.8*delta
        delta=2*delta;
    end
elseif rho<0.1 
    delta=delta/2;
end
if abs(s'*(y-B*s))>=r*norm(s)*norm(y-B*s)  %����B
B=B+(y-B*s)*(y-B*s)'/((y-B*s)'*s);
end
k=k+1;
end
end