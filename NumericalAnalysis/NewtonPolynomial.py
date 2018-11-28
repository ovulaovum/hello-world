# _*_ coding:utf-8 _*_
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x=sp.symbols('x')

def NewtonPolynomial(xk,yk):       #求牛顿插值多项式         
    x=sp.symbols('x')
    def DifferenceQuotient (xk,yk):    #求差商，返回一个list，包含f[x0]，f[x0，x1]到f[x0，...，xn]
        DiffQuo=yk                                      
        k=len(xk)-1
        j=0              
        i=0
        while i <k:
            j = k
            while j > i:
                if i == 0:
                    DiffQuo[j]=((yk[j]-yk[j-1])/(xk[j]-xk[j-1]))
                else:
                    DiffQuo[j]=(DiffQuo[j]-DiffQuo[j-1])/(xk[j]-xk[j-1-i])
                j -= 1
            i += 1
        return(DiffQuo)
    DiffQuo=DifferenceQuotient(xk, yk)
    g=1                                 
    f=0                          
    for i in range(len(xk)):      #迭代计算多项式   
        f=DiffQuo[i]*g+f
        g=(x-xk[i])*g
    return f                        

def Draw(f,a=-1.0,b=1.0,n=1001):    #画图
    d=np.linspace(a,b,n)
    x=sp.symbols('x')
    y=[]
    for t in d:
        y.append(f.subs(x,t))
    plt.figure()
    plt.plot(d,y)
    plt.show()
    return 0
Draw(NewtonPolynomial(np.linspace(-1,1,11), np.linspace(-1,1,11)))