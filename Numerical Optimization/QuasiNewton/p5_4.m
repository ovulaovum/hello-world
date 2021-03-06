x0=[-1.2 1
    0 0
    0.5 0.5
    2 2
    -1 -1];
f=@Rosenbrock;
g=@Rosenbrockg;
eps=10^(-5);
for i=1:5
    %[x,k]=DFP(x0(i,:)',eps,f,g);
    %[x,k]=BFGS(x0(i,:)',eps,f,g);
    [x,k]=SR1(x0(i,:)',eps,f,g);
end
function y = Rosenbrock(x)
y = (1-x(1))^2+100*(x(2)-x(1)^2)^2;
end
function y = Rosenbrockg(x)
y = [-2*(1-x(1))-400*x(1)*(x(2)-x(1)^2)
    200*(x(2)-x(1)^2)];
end
%所有方法从五个初始点开始迭代都收敛到(1,1)
%DFP方法的迭代次数分别为44 24 18 45 30，平均迭代次数为32.2
%初始化H为y'*s/(y'*y)*I常常使平均迭代次数少量增加
%任何情况下都更新H的迭代步数是最少的，但是有些参数下会收敛过慢
%BFGS方法的迭代次数分别为22 20 16 42 20，平均迭代次数为24
%SR1方法的迭代次数分别为57 35 31 15 40，平均迭代次数为35.6
%DFP跳过更新在-1 -1处需要908次迭代，而重置只需要30次