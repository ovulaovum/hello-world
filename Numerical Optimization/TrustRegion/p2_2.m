deltah=1; %最大半径
x0=[2,0]'; %初始点
delta0=0.2; %初始半径
eta=0.001; %最小预测比
eps=10^(-15); %精度
i=0;%记录迭代次数
x=x0;
delta=delta0;
[~,g,G]=m(x(1),x(2));
while norm(g)>eps
    [~,g,G]=m(x(1),x(2));
    p=dogleg(g,G,delta);
    rho=(Rosenbrock(x)-Rosenbrock(x+p))/(-g'*p-1/2*p'*G*p);
    if rho<1/4
        delta=1/4*delta;
    else
        if rho>3/4 && norm(p)==delta
            delta=min(2*delta,deltah);
        end
    end
    if rho>eta
        x=x+p;
    end
    i=i+1;
end
i
x