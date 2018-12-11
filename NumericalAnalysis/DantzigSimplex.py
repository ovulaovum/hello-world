# -*- coding: utf-8 -*-
import numpy as np
def Simplex(IB,IN,A,b,x,c):
#指标基IB,IN,约束A,b,初始解x,检验数r,目标函数z=cx,均为为np.mat类型,向量应为列向量
    m=len(IB)
    n=len(x)
    #计算初始参数
    z=c[IB].T*x[IB]
    B=((A.T)[IB]).T
    N=((A.T)[IN]).T 
    Br=B.I                                      #单独维护B^-1
    y=Br*A
    r=c.T-c[IB].T*y
    #def E(k,t,y):
    #    E=np.mat(np.eye(m))
    #    E[:,k]=-y[:,t]/y[k,t]
    #    E[k,t]=1/y[k,t]
    #    return E
    while not np.all(r[0,IN]>=0):               #判断是否为最优解
        t=np.argmin(r[0,IN])                    #确定列主元指标
        y[:,t]=Br*A[:,t]
        if all(y[:,t]<=0):                      #判断是否有下界
            print('无下界')
            return 0
        else:
            k=np.argmin([x[IB][i]/y[i,t] for i in range(m)])  #确定行主元指标
            a=x[IB][k]/y[k,t]
            x[IB]=x[IB]-y[:,t]*a                #更新x
            x[t]=a
            z=z+a*r[0,t]                        #更新目标函数
            v=np.where(IN==t)                
            IN[v]=IB[k]
            IB[k]=t                             #交换基指标
            N=((A.T)[IN]).T                     #更新N
            #Br=E(k,t,y)*Br                      #更新B^-1
            B=((A.T)[IB]).T 
            Br=B.I
            r[0,IN]=c[IN].T-(N.T*Br.T*c[IB]).T  #更新检验数
    return x,z[0,0]    

A=np.mat([[1.,1.,1.,0.],[2.,0.5,0.,1.]])
b=np.mat([5.,8.])
b=b.T
c=np.mat([-4.,-2.,0.,0.])
c=c.T
x=np.mat([0.,0.,5.,8.])
x=x.T
IB=np.array([2,3])
IN=np.array([0,1])
print(Simplex(IB,IN,A,b,x,c))

    