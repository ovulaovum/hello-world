function ps = dogleg(g,G,delta)
%ÕÛÏß·¨
%syms t
pB = -G^(-1)*g;
pU = -((g'*g)/(g'*G*g))*g;
if delta >= norm(pB)
    ps = pB;
elseif delta <= norm(pU)
    ps = pU/norm(pU)*delta;
else
    %tau = vpasolve(norm(pU+(t-1)*(pB-pU))^2 == delta^2,t);
    equ=[norm(pB-pU)^2,-2*(pB'*pB)+6*pB'*pU-4*(pU'*pU),4*(pU'*pU)-4*pB'*pU+(pB'*pB)-delta^2];
    tau=roots(equ);
    ps = pU+(tau(1)-1).*(pB-pU);
end
end