# _*_ coding:utf-8 _*_
import numpy as np
def getA(n):
    A=np.eye(n)
    A=A+np.mat(np.tril(-np.ones(n),-1))
    return A

def HilbertMatrix(n):
    H=np.mat(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            H[i,j]=1/(i+j+3)
    return H

def cond(A):
    def RowSumNorm(A):
        A=np.abs(A)    
        B=A.sum(axis=1)
        return np.max(B)
    return RowSumNorm(A)*RowSumNorm(A.I)

def getb(A):
    n=len(A)
    x=np.random.rand(n,1)
    return x,A*x

def PartialPivotingLU(A):
    n=len(A)
    p=[i for i in range(n)]  #记录排列阵
    for i in range(n-1):
        a=A[i,i]
        for j in range(i+1,n):  #选取列中最大元素
            if abs(a) < abs(A[j,i]):
                a=A[j,i]
                p[i]=j
        for k in range(n):
            a=A[p[i],k]
            A[p[i],k]=A[i,k]
            A[i,k]=a
        if A[i,i]!=0:
            for j in range(i+1,n):
                A[j,i]=A[j,i]/A[i,i]
            for j in range(i+1,n):
                for k in range(i+1,n):
                    A[j,k]=A[j,k]-A[j,i]*A[i,k]
        else:
            print('矩阵奇异')
            return 0
    return A,p

def PartialPivotingSolve(A,b):
    A=A.astype(np.float64)
    b=b.astype(np.float64)
    if np.shape(b)[0]==1:
        b=b.T
    A,p=PartialPivotingLU(A)
    n=len(A)
    for i in range(n-1):
        b[[i,p[i]]]=b[[p[i],i]]
    for i in range (1,n):
        for k in range(i):
            b[i]=b[i]-A[i,k]*b[k]
    for i in range(n):
        b[n-1-i]=b[n-1-i]/A[n-1-i,n-1-i] 
        for k in range(i):
            b[n-1-i]=b[n-1-i]-A[n-1-i,n-1-k]*b[n-1-k]/A[n-1-i,n-1-i]        
    return b

def CompletePivotingLU(A):
    n=len(A)
    p=[i for i in range(n)]  #记录排列阵
    q=[i for i in range(n)]
    for i in range(n-1):
        a=A[i,i]
        for j in range(i,n):  #选取子矩阵中最大元素
            for k in range(i,n):
                if abs(a) < abs(A[j,k]):
                    a=A[j,k]
                    p[i]=j
                    q[i]=k
        for j in range(n):
            a=A[p[i],j]
            A[p[i],j]=A[i,j]
            A[i,j]=a
        for j in range(n):
            a=A[j,q[i]]
            A[j,q[i]]=A[j,i]
            A[j,i]=a
        for j in range(i+1,n):
            A[j,i]=A[j,i]/A[i,i]
        for j in range(i+1,n):
            for k in range(i+1,n):
                A[j,k]=A[j,k]-A[j,i]*A[i,k]
    return A,p,q
    
def CompletePivotingSolve(A,b):
    A=A.astype(np.float64)
    b=b.astype(np.float64)
    if np.shape(b)[0]==1:
        b=b.T
    A,p,q=CompletePivotingLU(A)
    n=len(A)
    for i in range(n-1):
        b[[i,p[i]]]=b[[p[i],i]]
    for i in range (1,n):
        for k in range(i):
            b[i]=b[i]-A[i,k]*b[k]
    for i in range(n):
        b[n-1-i]=b[n-1-i]/A[n-1-i,n-1-i] 
        for k in range(i):
            b[n-1-i]=b[n-1-i]-A[n-1-i,n-1-k]*b[n-1-k]/A[n-1-i,n-1-i]        
    for i in range(1,n+1):
        b[[n-i,q[n-i]]]=b[[q[n-i],n-i]]
    return b
H=HilbertMatrix(12)
A=getA(60)
print(cond(H))
print(cond(A))
BackErrp=0
ForErrp=0
BackErrc=0
ForErrc=0
BackErrp2=0
ForErrp2=0
BackErrc2=0
ForErrc2=0
for i in range(10):
    y,c=getb(H)
    y_p=PartialPivotingSolve(H,c)
    y_c=CompletePivotingSolve(H,c)
    BackErrp+=np.linalg.norm(c-H*y_p,2)/10
    ForErrp+=np.linalg.norm(y-y_p,2)/10 
    BackErrc+=np.linalg.norm(c-H*y_c,2)/10  
    ForErrc+=np.linalg.norm(y-y_c,2)/10     
    A=getA(60)
    x,b=getb(A)
    x_p=PartialPivotingSolve(A,b)
    x_c=CompletePivotingSolve(A,b)
    BackErrp2+=np.linalg.norm(b-A*x_p,2)/10
    ForErrp2+=np.linalg.norm(x-x_p,2)/10
    BackErrc2+=np.linalg.norm(b-A*x_c,2)/10
    ForErrc2+=np.linalg.norm(x-x_c,2)/10
print('BackErrp=',BackErrp)
print('ForErrp=',ForErrp)
print('BackErrc=',BackErrc)
print('ForErrc=',ForErrc)
print('BackErrp=',BackErrp2)
print('ForErrp=',ForErrp2)
print('BackErrc=',BackErrc2)
print('ForErrc=',ForErrc2)