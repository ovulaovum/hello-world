function [x,k] = BFGS(x,eps,f,g)
%BFGS ��ţ�ٷ�
%��������ʼ������x������eps��Ŀ�꺯��f���ݶ�g
n=length(x);%��ȡά��
gk=g(x);%��ʼ���ݶ�
fk=f(x);%��ʼ������ֵ
k=0;%��¼��������
I=eye(n,n);
H=I;%��ʼ��H

% %����һ������ʼ����ɭ���������H
% p=-gk;
% alpha=1;
% while f(x+alpha*p)>fk+10^(-4)*alpha*gk'*p
%     alpha=0.4*alpha;
% end
% s=alpha*p;
% y=g(x+s)-gk;
% H=y'*s/(y'*y)*I;

%��ʼ����
while norm(gk)>eps
if k>=10000
    k='maxiterations';%������������ֹ����
    break
end
p=-H*gk;
%armijo��������
alpha=1;
while f(x+alpha*p)>fk+10^(-4)*alpha*gk'*p
    alpha=0.42*alpha;
end
s=alpha*p;
x=x+s;
y=g(x)-gk;
if y'*s>10^(-10)  %y'*s��Сʱ����H
    rho=1/(y'*s);
    H=(I-rho*s*y')*H*(I-rho*y*s')+rho*(s*s');%����H
else
    H=I;
end
fk=f(x);
gk=g(x);
k=k+1;
end
end