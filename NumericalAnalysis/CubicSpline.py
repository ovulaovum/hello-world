# _*_ coding:utf-8 _*_
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
sp.init_printing(use_unicode=False, wrap_line=False, no_global=True)
x=sp.symbols('x')

def DifferenceQuotient3 (xk,yk):  #求f[xi-1,xi,xi+1]
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
        return(DiffQuo[2])

def CubicSpline (f,xk):    #求三次样条函数
    yk=[]
    n=len(xk)
    for i in range(n):    
        yk.append(f.subs(x,xk[i]))
    hk=[]
    for i in range(n-1):
        hk.append(xk[i+1]-xk[i])
    def GetMatrix(xk):    #计算系数矩阵
        hk=[]
        n=len(xk)
        for i in range(n-1):
            hk.append(xk[i+1]-xk[i])
        mat=np.zeros((n,n))
        for i in range(n):
            mat[i,i]=2
        for i in range(n-1):
            if i == 0:
                mat[i,i+1]=1
                mat[i+1,i]=hk[i]/(hk[i+1]+hk[i])     
            elif i==n-2:
                mat[i,i+1]=hk[i]/(hk[i]+hk[i-1])    
                mat[i+1,i]=1  
            else:
                mat[i,i+1]=hk[i]/(hk[i]+hk[i-1])    
                mat[i+1,i]=hk[i]/(hk[i+1]+hk[i])    
        return mat
    A=np.mat(GetMatrix(xk))
    yn=sp.diff(f,x).subs(x,xk[n-1])
    y0=sp.diff(f,x).subs(x,xk[0])
    b=[6*DifferenceQuotient3([xk[i-1],xk[i],xk[i+1]], [yk[i-1],yk[i],yk[i+1]]) for i in range (1,n-1)]
    b.append(6/hk[n-2]*(yn-(yk[n-1]-yk[n-2])/hk[n-2]))
    b.insert(0,6/hk[0]*((yk[1]-yk[0])/hk[0]-y0))
    b=np.array(b)
    b=b.astype(np.float64)
    m=b*A.I        #解出Mi
    S=[]
    for i in range(n-1):
        S.append(((m[0,i]*(xk[i+1]-x)**3)/6)+((m[0,i+1]*(x-xk[i])**3)/6)+(yk[i]-m[0,i]*hk[i]*hk[i]/6)*(xk[i+1]-x)/hk[i]+(yk[i+1]-m[0,i+1]*hk[i]*hk[i]/6)*(x-xk[i])/hk[i])
    return (S)
    

f=1/(1+25*x**2)
x10=np.linspace(-1.0,1.0,11)
x20=np.linspace(-1.0,1.0,21)
s1=CubicSpline(f, x10)
s2=CubicSpline(f, x20)


dotx1=[]
doty1=[]
dotx2=[]
doty2=[]
for i in range(10):
    for j in np.linspace(x10[i],x10[i+1],101):
        dotx1.append(j)
        doty1.append(s1[i].subs(x,j))
for i in range(20):
    for j in np.linspace(x20[i],x20[i+1],101):
        dotx2.append(j)
        doty2.append(s2[i].subs(x,j))
xx=np.linspace(-1.0,1.0,1001)
yy=[]
yyy=[]
for i in range(1001):
    yyy.append(f.subs(x,xx[i]))    #原函数
plt.figure()
plt.plot(dotx1,doty1)
plt.plot(dotx2,doty2)
plt.plot(xx,yyy)
plt.show()
