%initialize
n=200;
v=rand(n,1);
U=rand(n);
[Q,R]=qr(U);
M=Q'*diag(v)*Q;  %n*n
a=n/50;
A=rand(a,n);
B=rand(a,n);
lambda=randn(a,1);
sigma=rand(1);
mu=[rand(a/2,1);zeros(a/2,1)];
delta=1;
p=rand(n,1);
p=p/norm(p);
beta=A*p;  %a*1
gamma=B*p+[zeros(a/2,1);rand(a/2,1)];  %a*1
q=M*p+sigma*p+A'*lambda+B'*mu;  %n*1
q=-q;
p0=p;
esp=10^(-4);
y=zeros(a,1);
z=zeros(n,1);
rho=1;
lambda1=zeros(a,1);
lambda2=zeros(a,1);
lambda3=zeros(n,1);
%iterate
E=M+rho*(A'*A+B'*B+eye(n));
while max([norm(M*p+q-A'*lambda1-B'*lambda2-lambda3),norm(A*p-beta,inf),norm(B*p+y-gamma,inf),norm(p-z,inf)])>esp
    %p=argmin L(p,y,z,lambda1,lambda2,lambda3)
    p=E\(-(q-A'*lambda1-B'*lambda2-lambda3-rho*A'*beta+rho*B'*(y-gamma)-rho*z));    
    %y=argmin L(p,y,z,lambda1,lambda2,lambda3),y>=0
    y=-(rho*(B*p-gamma)-lambda2)./rho;
    for i=1:a
        if y(i)<0
            y(i)=0;
        end
    end
    %z=argmin L(p,y,z,lambda1,lambda2,lambda3),norm(z)<=delta
    u=lambda3-rho*p;
    if norm(u)^2<=rho^2*delta
        z=-u./rho;
    else 
        z=-sqrt(delta)/norm(u).*u;
    end
    %update lambda1,2,3
    lambda1=lambda1-rho*(A*p-beta);
    lambda2=lambda2-rho*(B*p+y-gamma);
    lambda3=lambda3-rho*(p-z);
end
%validate
fprintf('norm(lambda1+lambda)=%e\n',norm(lambda1+lambda));
fprintf('norm(lambda2+mu)=%e\n',norm(lambda2+mu));
fprintf('norm(lambda3+sigma*p0)=%e',norm(lambda3+sigma*p0));