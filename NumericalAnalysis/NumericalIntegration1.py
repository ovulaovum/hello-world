# _*_ coding:utf-8 _*_
import numpy as np
import sympy as sp
x=sp.symbols('x')
f2=sp.symbols('f2')
f4=sp.symbols('f4')
def CompositeTrapezoidal(f,a,b,n,error=0):
    h=(b-a)/n
    yk=[f.subs(x,a+i*h) for i in range(n+1)]
    T=0
    for i in range(n):
        T=T+(b-a)/2/n*(yk[i]+yk[i+1])
    if error:
        R=-(b-a)*x**2/12*f2
        return R
    return T
def CompositeSimpson(f,a,b,n,error=0):
    h=(b-a)/n
    yk=[f.subs(x,a+i*h) for i in range(n+1)]
    yk2=[f.subs(x,a+i*h+h/2) for i in range(n)]
    S=h/6*(yk[0]+yk[n]+4*yk2[0])
    for i in range(1,n):
        S=S+h/6*(4*yk2[i]+2*yk[i])
    if error:
        R=-(b-a)/180/16*x**4*f4    
        return R
    return S
def Romberg(f,a,b,e):
    k=1
    h=(b-a)/2
    T=np.zeros((20,20))
    T[0,0]=h*(f.subs(x,a)+f.subs(x,b))
    T[0,1]=T[0,0]/2+h*f.subs(x,a+h)
    T[1,0]=(4*T[0,1]-T[0,0])/3
    while abs(T[k,0]-T[k-1,0])>e: 
        k=k+1
        h=h/2
        T[0,k]=T[0,k-1]/2
        for i in range(1,2**(k-1)):
            T[0,k]=T[0,k]+h*f.subs(x,a+2*i*h+h)
        for j in range(1,k+1):
            T[j,k-j]=(4**j*T[j-1,k+1-j]-T[j-1,k-j])/(4**j-1)
    return T[k,0]
def AdaptiveSimpson(f,a,b,e):
    def Recursion(a,b,e,A):
        c=(a+b)/2
        B=CompositeSimpson(f,a,c,1)
        C=CompositeSimpson(f,c,b,1)
        if abs(B+C-A) <= 15*e:
            return B+C+(B+C-A)/15;
        return Recursion(a,c,e/2,B)+Recursion(c,b,e/2,C);  
    return Recursion(a,b,e,CompositeSimpson(f, a, b, 1))
f=x**(1/2)*sp.log(x)
print('x**(1/2)*ln(x)在(0,1)上积分，复合梯形公式计算结果为(n=100)' ,CompositeTrapezoidal(f, 0.0001, 1, 100))
print('截断误差函数为', CompositeTrapezoidal(f, 0.0001, 1, 100, 1),',其中f2为(0,1)上某一点的二阶导数值')
print('与积分精确值的误差为',CompositeTrapezoidal(f, 0.0001, 1, 100)+4/9)
print('x**(1/2)*ln(x)在(0,1)上积分，复合辛普森公式计算结果为(n=100)' ,CompositeSimpson(f, 0.0001, 1, 100))
print('截断误差函数为', CompositeSimpson(f, 0.0001, 1, 100, 1),',其中f4为(0,1)上某一点的四阶导数值')
print('与积分精确值的误差为',CompositeSimpson(f, 0.0001, 1, 100,)+4/9)
print('可知复合辛普森公式的精度更高')
print('误差的绝对值是单调减函数，h越小误差越小，不存在最小的h使得精度无法被改善')
print('预设精度为10^-2时，龙贝格算法计算结果为' ,Romberg(f,0.0001,1,0.01))
print('预设精度为10^-4时，自适应辛普森算法计算结果为',AdaptiveSimpson(f, 0.001, 1, 0.0001))