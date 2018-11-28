# _*_ coding:utf-8 _*_
import numpy as np
import sympy as sp
x=sp.symbols('x')
y=sp.symbols('y')
def CompositeSimpson(f,x,a,b,n):
    h=(b-a)/n
    yk=[f.subs(x,a+i*h) for i in range(n+1)]
    yk2=[f.subs(x,a+i*h+h/2) for i in range(n)]
    S=h/6*(yk[0]+yk[n]+4*yk2[0])
    for i in range(1,n):
        S=S+h/6*(4*yk2[i]+2*yk[i])
    return S
def Gauss4(f,x,a,b):
    xk=[-0.9851798,-0.5384693,0,0.5384693,0.9851798]
    Ak=[0.2369269,0.4786287,0.5688889,0.4786287,0.2369269]
    yk=[f.subs(x,xk[i]) for i in range(5)]
    I=0
    for i in range(5):
        I=I+yk[i]*Ak[i]    
    return I
f=sp.exp(-x*y)
g=sp.exp(-(x+1)*(y+1)/4)/4
print('exp(-x*y)在[0,1]*[0,1]上积分，复合辛普森公式(n=4)计算结果为',CompositeSimpson(CompositeSimpson(f, x, 0, 1, 4), y, 0, 1, 4))
print('exp(-x*y)在[0,1]*[0,1]上积分，高斯求积公式(n=4)计算结果为',float(Gauss4(Gauss4(g, x, -1, 1), y, -1, 1)))
print('exp(-x*y)在1/4单位圆上积分，复合辛普森公式(n=4)计算结果为',CompositeSimpson(CompositeSimpson(f, x, 0, (1-y**2)**(1/2), 4),y,0,1,4))