import numpy as np
e=np.exp
def f(x,y):             #右端函数
    f=(x*x+y*y)*e(x*y)
    return -f

def FiveStencil(n,f):   #五点差分格式
    h=1/(n+1)
    A=4*np.eye(n*n)-np.eye(n*n,k=1)-np.eye(n*n,k=-1)-np.eye(n*n,k=n)-np.eye(n*n,k=-n)  #系数矩阵
    A=np.mat(A)
    for i in range(1,n):
        A[i*n-1,i*n]=0
        A[i*n,i*n-1]=0  
    b=np.zeros((n*n,1))
    b=np.mat(b)
    for k in range(n*n):
        i=(k)%n+1
        j=(k)//n+1
        b[k]=h*h*f(i*h,j*h) 
        if i==1:        #边值条件
            b[k]+=1
        elif i==n:
            b[k]+=e(j*h)
        if j==1:
            b[k]+=1
        elif j==n:
            b[k]+=e(i*h)
    return A,b

def JacobiMethod(A,b,x0,eps):
    n=len(b)
    D=np.diag(np.diag(A))
    D=np.mat(D)
    L=-np.tril(A,-1)
    U=-np.triu(A,1)
    B=D.I*(L+U)
    g=D.I*b
    k=0
    x=x0
    xx=x+2*eps
    while np.linalg.norm(x-xx,np.inf)>eps:
        xx=x
        x=B*x+g
        if k==0:
            xf=x
        k+=1
    return x,xf,k

def SOR(A,b,x0,w,eps):
    n=len(b)
    D=np.diag(np.diag(A))
    D=np.mat(D)
    L=np.mat(-np.tril(A,-1))
    U=np.mat(-np.triu(A,1))
    Lw=(D-w*L).I*((1-w)*D+w*U)
    g=w*(D-w*L).I*b
    k=0
    x=x0
    xt=x+2*eps
    while np.linalg.norm(x-xt,np.inf)>eps:
        xt=x
        x=Lw*x+g
        if k==0:
            xf=x
        k+=1
    return x,xf,k

def ConjugateGradient(A,b,x0,eps):
    x=x0
    n=len(b)
    r=b-A*x
    k=0
    rho=r.T*r
    rhot=0
    beta=0
    while rho>eps**2*np.linalg.norm(b,np.inf)**2 and k<n:
        k+=1
        if k==1:
            p=r
            xf=x
        else:
            beta=rho[0,0]/rhot[0,0]
            p=r+beta*p
        w=A*p
        a=rho/(p.T*w)
        x=x+a[0,0]*p
        r=r-a[0,0]*w
        rhot=rho
        rho=r.T*r
    return x,xf,k

def xx(n):
    h=1/(n+1)
    xx=np.mat(np.zeros((n*n,1)))
    for k in range(n*n):
        i=(k)%n+1
        j=(k)//n+1
        xx[k]=e(i*h*j*h)
    return xx

A,b=FiveStencil(10,f)
x0=np.ones((100,1))
#x是解，xf是第一次迭代的解，Err是误差
xj,xfj,kj=JacobiMethod(A,b,x0,10**(-5))
x1,xf1,k1=SOR(A,b,x0,1,10**(-5))
x125,xf125,k125=SOR(A,b,x0,1.25,10**(-5))
x15,xf15,k15=SOR(A,b,x0,1.5,10**(-5))
x175,xf175,k175=SOR(A,b,x0,1.75,10**(-5))
xcg,xfcg,kcg=ConjugateGradient(A,b,x0,10**(-5))
Errj=np.linalg.norm(xj-xx(10),np.inf)
Errw1=np.linalg.norm(x1-xx(10),np.inf)
Errw125=np.linalg.norm(x125-xx(10),np.inf)
Errw15=np.linalg.norm(x15-xx(10),np.inf)
Errw175=np.linalg.norm(x175-xx(10),np.inf)
Errcg=np.linalg.norm(xcg-xx(10),np.inf)
print('Jacobi法计算的误差和迭代次数分别为',Errj,kj)
print('SOR法取w=1计算的误差和迭代次数分别为',Errw1,k1)
print('SOR法取w=1.25计算的误差和迭代次数为',Errw125,k125)
print('SOR法取w=1.5计算的误差和迭代次数为',Errw15,k15)
print('SOR法取w=1.75计算的误差和迭代次数为',Errw175,k175)
print('共轭梯度法计算的误差和迭代次数为',Errcg,kcg)        