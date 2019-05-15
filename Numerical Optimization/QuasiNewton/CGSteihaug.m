function p = CGSteihaug(eps,delta,r,B)
%CGSTEIHAUG 求解信赖域子问题
%参数：精度eps，半径delta，梯度r，海森近似阵B
n=length(r);
d=-r;
z=zeros(n,1);
if norm(r)<eps
    p=0;
end
for i=1:n
if d'*B*d<=0
    eq=[d'*d,2*z'*d,z'*z-delta^2];
    root=roots(eq);
    if root(1)>=0
        p=z+root(1)*d;
    else
        p=z+root(2)*d;
    end
    break
end
alpha=r'*r/(d'*B*d);
if norm(z+alpha*d)>=delta
    eq=[d'*d,2*z'*d,z'*z-delta^2];
    root=roots(eq);
    if root(1)>=0
        p=z+root(1)*d;
    else
        p=z+root(2)*d;
    end
    break
end
z=z+alpha*d;
beta=(r+alpha*B*d)'*(r+alpha*B*d)/(r'*r);
r=r+alpha*B*d;
if norm(r)<eps
    p=z;
    break
end
d=-r+beta*d;
end
end