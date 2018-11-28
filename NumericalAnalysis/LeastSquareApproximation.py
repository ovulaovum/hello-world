# _*_ coding:utf-8 _*_
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
x=sp.symbols('x')
xx=np.linspace(-1,1,11)
f=1/(1+25*x**2)
def Legendre(n):
    if n==0:
        return 1+0*x
    else:
        return sp.diff((x**2-1)**n,x,n)/(2**n)/np.math.factorial(n)
def product(f,g,xx):
    n=len(xx)
    p=0
    for i in range(n):
        p=p+f.subs(x,xx[i])*g.subs(x,xx[i])
    return p 
def LSA(f,xx,n):
    s=0
    for i in range(n):
        p=Legendre(i)
        s=s+product(p,f,xx)/product(p,p,xx)*p
    return s
xxx=np.linspace(-1,1,1001)
y1=[]
y2=[]
for i in range(1001):
    y1.append(f.subs(x,xxx[i]))
    y2.append(LSA(f,xx,3).subs(x,xxx[i]))
plt.figure()
plt.plot(xxx,y1)
plt.plot(xxx,y2)
plt.show()

