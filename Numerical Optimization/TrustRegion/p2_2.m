deltah=1; %���뾶
x0=[2,0]'; %��ʼ��
delta0=0.2; %��ʼ�뾶
eta=0.001; %��СԤ���
eps=10^(-15); %����
i=0;%��¼��������
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