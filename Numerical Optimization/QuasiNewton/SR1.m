function [x,k] = SR1(x,eps,f,g)
%SR1 拟牛顿法
%参数：初始迭代点x，精度eps，目标函数f，梯度g
n=length(x);%获取维数
gk=g(x);%初始化梯度
delta=0.2; %初始半径
eta=0.01; %最小预测比
r=10^(-8);%判断跳过更新的系数
k=0;%记录迭代次数
B=eye(n,n);%初始化H
% %迭代一步，初始化海森近似阵的逆H
% p=-gk;
% alpha=1;
% while f(x+alpha*p)>fk+10^(-4)*alpha*gk'*p
%     alpha=0.4*alpha;
% end
% s=alpha*p;
% y=g(x+s)-gk;
% H=y'*s/(y'*y)*eye(n,n);

%开始迭代
while norm(gk)>eps
if k>=10000
    k='maxiterations';%收敛过慢则终止迭代
    break
end
s=CGSteihaug(eps,delta,gk,B);
y=g(x+s)-gk;
rho=(f(x)-f(x+s))/(-gk'*s-1/2*s'*B*s);
if rho>eta  %更新x
    x=x+s;
end
gk=g(x);
if rho>3/4  %更新半径
    if norm(s)>0.8*delta
        delta=2*delta;
    end
elseif rho<0.1 
    delta=delta/2;
end
if abs(s'*(y-B*s))>=r*norm(s)*norm(y-B*s)  %更新B
B=B+(y-B*s)*(y-B*s)'/((y-B*s)'*s);
end
k=k+1;
end
end