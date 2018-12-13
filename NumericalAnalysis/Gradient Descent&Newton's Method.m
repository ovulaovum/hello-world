n=100;
esp=10^(-7);
x=defx(n);
f=Rosenbrock(n);
%avoid symbolic calculation
F=matlabFunction(f,'Vars',{x});            
G=matlabFunction(Grad(f,n),'Vars',{x});
H=matlabFunction(Hessian(f,n),'Vars',{x});
%initialize parameters
x0=zeros(1,n);
GradientDescent(F,G,x0,esp);
%n=20 esp=10^(-5) t=1.027s esp=10^(-6) t=1.065s esp=10^(-7) t=1.060s
%n=100 esp=10^(-5) t=3.788s esp=10^(-6) t=3.813s esp=10^(-7) t=3.799s
nm=NewtonsMethod(G,H,x0,esp);
%n=20 esp=10^(-5) t=11.201s esp=10^(-6) t=11.154s esp=10^(-7) t=11.065s
%n=100 esp=10^(-5) t=263.407s esp=10^(-6) t=264.325s esp=10^(-7) t=265.154s
%Creating Hessian matirx costs about 250s
function x=defx(n)   %symbolic vector x
x=sym('x',[1,n]);
end
function a=Armijo(F,G,xk,dk)
a=1;
p=0.7;
c=0.001;
while any(F(xk+a*dk).'>F(xk).'+c*a*G(xk)*dk.')
a=p*a;
end
end 
function f=Rosenbrock(n)
f=0;
x=defx(n);
for i=1:n-1
f=f+100*sum((x(i+1)-x(i)^2)^2+(1-x(i))^2);
end
end
function g=Grad(f,n)
g=sym(zeros(1,n));
x=defx(n);
for i=1:n
    g(i)=diff(f,x(i));
end
end
function H=Hessian(f,n)
H=sym(zeros(n));
x=defx(n);
for i=1:n
    for j=1:n
        H(i,j)=diff(diff(f,x(i)),x(j));
    end
end
end
function xk=GradientDescent(F,G,x0,esp)
xk=x0;
dk=-G(xk);
while norm(dk)>esp
ak=Armijo(F,G,xk,dk);
xk=xk+ak*dk;
dk=-G(xk);
end
end
function xk=NewtonsMethod(G,H,x0,esp)
xk=x0;
pk=-G(xk)*H(xk)^(-1).'; %Newton step
while norm(pk)>esp
    xk=xk+pk;
    pk=-G(xk)*H(xk)^(-1).';
end
end