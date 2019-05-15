function [x,k] = CG(A,b,x0,esp)
%CG 线性共轭梯度法求解线性系统Ax=b，初始点x0，精度esp
x=x0;
r2=A*x-b;
p=-r2;
k=0;
while norm(r2)>esp
    alpha=r2'*r2/(p'*A*p);
    x=x+alpha*p;
    r1=r2;
    r2=r2+alpha*A*p;
    beta=r2'*r2/(r1'*r1);
    p=-r2+beta*p;
    k=k+1;
end
end