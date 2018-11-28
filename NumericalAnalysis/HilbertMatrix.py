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
for i in range(2,7):
    print('%d阶希尔伯特矩阵的条件数为%d'%(i,cond(HilbertMatrix(i))))
def b(n):
    x=np.mat([1 for i in range(n)])
    return x*HilbertMatrix(n)

def LUSolve(A,b):
    A=A.astype(float)
    b=b.astype(float)
    b=b.T
    n=len(A)
    p=[i for i in range(n)]  #排列阵
    for i in range(n-1):
        a=A[i,i]
        for j in range(i+1,n):
            if abs(a) < abs(A[j,i]):
                a=A[j,i]
                p[i]=j
        for k in range(n):
            a=A[p[i],k]
            A[p[i],k]=A[i,k]
            A[i,k]=a
        for j in range(i+1,n):
            A[j,i]=A[j,i]/A[i,i]
        for j in range(i+1,n):
            for k in range(i+1,n):
                A[j,k]=A[j,k]-A[j,i]*A[i,k]
    #LU分解完成
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
    
    
for s in range(2,5):
    print('r_%d='%(s),b(s).T-HilbertMatrix(s)*LUSolve(HilbertMatrix(s), b(s)))
    print('Δx_%d='%(s),LUSolve(HilbertMatrix(s),b(s))-[[1] for i in range(s)])
print('n=3时x^bar就一位有效数字都没有了')