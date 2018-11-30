# _*_ coding:utf-8 _*_
import numpy as np

def HilbertMatrix(n):
    H=np.mat(np.zeros((n,n)))
    for i in range(n):
        for j in range(n):
            H[i,j]=1/(i+j+1)
    return H
def cond(A):
    def RowSumNorm(A):
        A=np.abs(A)    
        B=A.sum(axis=1)
        return np.max(B)
    return RowSumNorm(A)*RowSumNorm(A.I)

#for i in range(2,7):
#    print('%d阶希尔伯特矩阵的条件数为%d'%(i,cond(HilbertMatrix(i))))

def b(n):
    x=np.mat([1 for i in range(n)])
    return x*HilbertMatrix(n)

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
    A=A.astype(float)
    b=b.astype(float)
    b=b.T
    A,p=PartialPivotingLU(A)
    n=len(A)
    for i in range(n-1):
            a=b[i]
            b[i]=b[p[i]]
            b[p[i]]=a
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
    A=A.astype(float)
    b=b.astype(float)
    b=b.T
    A,p,q=CompletePivotingLU(A)
    n=len(A)
    for i in range(n-1):
        a=b[i]
        b[i]=b[p[i]]
        b[p[i]]=a
    for i in range (1,n):
        for k in range(i):
            b[i]=b[i]-A[i,k]*b[k]
    for i in range(n):
        b[n-1-i]=b[n-1-i]/A[n-1-i,n-1-i] 
        for k in range(i):
            b[n-1-i]=b[n-1-i]-A[n-1-i,n-1-k]*b[n-1-k]/A[n-1-i,n-1-i]        
    for i in range(1,n+1):
        a=b[n-i]
        b[n-i]=b[q[n-i]]
        b[q[n-i]]=a
    return b

