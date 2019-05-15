M=rand(100);
[Q,R]=qr(M);
A=diag(1.2:0.2:21);%特征值均匀分布
A=Q'*A*Q;
v=[1.01:0.01:1.5,9.01:0.01:9.5];%特征值集中分布
B=diag(v);
B=Q'*B*Q;
b=ones(100,1);
x=zeros(100,1);
esp=10^(-5);
[~,ka]=CG(A,b,x,esp);
[~,kb]=CG(B,b,x,esp);
% ka =
% 
%     29
% 
% 
% kb =
% 
%     12