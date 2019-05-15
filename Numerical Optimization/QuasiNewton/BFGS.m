function [x,k] = BFGS(x,eps,f,g)
%BFGS 拟牛顿法
%参数：初始迭代点x，精度eps，目标函数f，梯度g
n=length(x);%获取维数
gk=g(x);%初始化梯度
fk=f(x);%初始化函数值
k=0;%记录迭代次数
I=eye(n,n);
H=I;%初始化H

% %迭代一步，初始化海森近似阵的逆H
% p=-gk;
% alpha=1;
% while f(x+alpha*p)>fk+10^(-4)*alpha*gk'*p
%     alpha=0.4*alpha;
% end
% s=alpha*p;
% y=g(x+s)-gk;
% H=y'*s/(y'*y)*I;

%开始迭代
while norm(gk)>eps
if k>=10000
    k='maxiterations';%收敛过慢则终止迭代
    break
end
p=-H*gk;
%armijo步长规则
alpha=1;
while f(x+alpha*p)>fk+10^(-4)*alpha*gk'*p
    alpha=0.42*alpha;
end
s=alpha*p;
x=x+s;
y=g(x)-gk;
if y'*s>10^(-10)  %y'*s过小时重置H
    rho=1/(y'*s);
    H=(I-rho*s*y')*H*(I-rho*y*s')+rho*(s*s');%更新H
else
    H=I;
end
fk=f(x);
gk=g(x);
k=k+1;
end
end