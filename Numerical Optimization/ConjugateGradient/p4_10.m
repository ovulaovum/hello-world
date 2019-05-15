x0=[-1.2,1]';
for esp=[10^(-5),10^(-6),10^(-7)]
%初始化
k=0;
x=x0;
g2=g(x);
p=-g2;
    while norm(g2)>esp
    alpha=1;
          while f(x+alpha*p)>f(x)+10^(-4)*alpha*g2'*p
              alpha=0.5*alpha;
          end
    x=x+alpha*p;%更新x
    g1=g2;%上一个迭代点的梯度
    g2=g(x);%更新梯度
    %beta=g2'*g2/(g1'*g1);%FR
    %beta=g2'*(g2-g1)/(g1'*g1);%PR
    beta=g2'*(g2-g1)/((g2-g1)'*p);%HS
    %beta=0;%GD
    p=-g2+beta*p;%更新步长
    k=k+1;
    end
x,k
end
function y = f(x)
y = (1-x(1))^2+100*(x(2)-x(1)^2)^2;
end
function y = g(x)
y = [-2*(1-x(1))-400*x(1)*(x(2)-x(1)^2)
    200*(x(2)-x(1)^2)];
end
%所有方法均能收敛到(1,1)
%FR方法在三个精度下的迭代次数分别为221,268,286
%PR方法在三个精度下的迭代次数分别为1735,1852,1954
%HS方法在三个精度下的迭代次数分别为73,79,86
%直接使用梯度下降法在三个精度下的迭代次数分别为10916,13756,16956