import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt

def check_len(l,n):
    if len(l) != n:
        print('length of input should be %d, not %d'% (n,len(l)))
    return len(l)-n

#f为方程右端，t，u是初值列表，h是步长，turn是迭代轮数，n是步数，phi是迭代格式
#f应实现方法cal()计算函数值，以及grad()计算导数值
def solve_ode(f,t0,u0,h,turn,n,phi):
    check_len(t0,n)
    check_len(u0,n)
    if n>1 and t0[1]-t0[0] != h:
        print('t is not consistent with h')
    t=t0[:]
    u=u0[:]
    for i in range(n-1,turn):
        u.append(phi(f,t[i-n+1:],u[i-n+1:],h))
        t.append(t[i]+h)
    return t,u

#画图，f是右端函数，t0是起始点列表，h是步长列表，turn是迭代轮数列表，c是精确解函数，n1是步数，phi1是迭代格式，
#n2是初始化格式步数，phi2是初始化格式，ini=0则不需要初始化，返回双对数收敛阶图像数据，n1>=n2
def show_conv(f,t0,h,turn,n1,phi1,c,n2=1,phi2=0,ini=0):
    x=[]
    y=[]
    for i in range(len(h)):
        print('h=',h[i])
        t0=[t0[0]+j*h[i] for j in range(n1)]
        u0=[c(t0[j]) for j in range(n1)]
        if ini > 0:
            _, u0=solve_ode(f,t0[:n2],u0[:n2],h[i],n1-1,n2,phi2)
        _, u=solve_ode(f,t0,u0,h[i],turn[i],n1,phi1)
        y.append(np.log(abs(u[-1]-c(t0[0]+h[i]*turn[i]))))
        x.append(np.log(h[i]))
    return x,y

#画图，f是右端函数，t0是起始点列表，h是步长，turn是迭代轮数，c是精确解函数，n1是步数，phi1是迭代格式，
#n2是初始化格式步数，phi2是初始化格式，ini=0则不需要初始化，返回解图像数据，n1>=n2
def draw_pic(f,t0,h,turn,n1,phi1,c,n2=1,phi2=0,ini=0):
    x=[]
    y=[]
    t0=[t0[0]+j*h for j in range(n1)]
    u0=[c(t0[j]) for j in range(n1)]
    if ini > 0:
        _, u0=solve_ode(f,t0[:n2],u0[:n2],h,n1-1,n2,phi2)
    t, u=solve_ode(f,t0,u0,h,turn,n1,phi1)
    return t, u

#newton法用于解非线性方程
def newton(f,g,x,eps):
    while abs(f(x))>eps:
        d=-1/g(x)*f(x)
        x=x+d
    return x

#以下是单步方法
def taylor2(f,t,u,h):
    return u[0]+h*(f.cal(t[0],u[0])+h/2*(1-2*u[0])*f.cal(t[0],u[0]))

def taylor3(f,t,u,h):
    return u[0]+h*(f.cal(t[0],u[0])+h/2*(1-2*u[0])*f.cal(t[0],u[0])+h*h/6*((1-2*u[0])*(1-2*u[0])*f.cal(t[0],u[0])-2*f.cal(t[0],u[0])*f.cal(t[0],u[0])))

def euler(f,t,u,h):
    return u[0]+h*f.cal(t[0],u[0])

def hidden_euler(f,t,u,h):
    def a(x):
        return x-u[0]-h*f.cal(t[0]+h,x)
    def b(x):
        return 1-h*f.grad(t[0]+h,x)
    return newton(a,b,u[0]+h*f.cal(t[0],u[0]),10**(-14))

def runge_kutta(f,t,u,h,a,b,c):
    k=[f.cal(t[0],u[0])]
    phi=c[0]*f.cal(t[0],u[0])
    for i in range(1, len(a)):
        temp=0
        for j in range(len(b[i])):
            temp+=b[i][j]*k[j]
        k.append(f.cal(t[0]+a[i]*h,u[0]+h*temp))
        phi+=c[i]*f.cal(t[0]+a[i]*h,u[0]+h*temp)
    return u[0]+h*phi

def runge_kutta4(f,t,u,h):
    a4=[0,1/2,1/2,1]
    b4=[[],[1/2],[0,1/2],[0,0,1]]
    c4=[1/6,1/3,1/3,1/6]
    return runge_kutta(f,t,u,h,a4,b4,c4)

def runge_kutta3(f,t,u,h):
    a3=[0,1/2,1]
    b3=[[],[1/2],[-1,2]]
    c3=[1/6,2/3,1/6]
    return runge_kutta(f,t,u,h,a3,b3,c3)

def runge_kutta2(f,t,u,h):
    a2=[0,1]
    b2=[[],[1]]
    c2=[1/2,1/2]
    return runge_kutta(f,t,u,h,a2,b2,c2)

#以下是多步方法
def multistep(f,t,u,h,alpha,beta):
    check_len(t,len(alpha)-1)
    check_len(u,len(beta)-1)
    z=0
    if beta[-1]!=0:
        def a(x):
            y=0
            for i in range(len(alpha)-1):
                y+=alpha[i]*u[i]
                y-=beta[i]*h*f.cal(t[i],u[i])
            return alpha[-1]*x-beta[-1]*h*f.cal(t[-1]+h,x)+y
        def b(x):
            y=0
            return alpha[-1]-beta[-1]*h*f.grad(t[-1]+h,x)
        z=newton(a,b,u[-1]+h*f.cal(t[-1],u[-1]),10**(-14))
    else:
        y=0
        for i in range(len(beta)-1):
            y+=beta[i]*h*f.cal(t[i],u[i])
            y-=alpha[i]*u[i]
        z=y/alpha[-1]
    return z

def adams_moulton4(f,t,u,h):
    check_len(t,3)
    check_len(u,3)
    return multistep(f,t,u,h,[0,0,-1,1],[1/24,-5/24,19/24,9/24])

def gear4(f,t,u,h):
    check_len(t,4)
    check_len(u,4)
    return multistep(f,t,u,h,[1/4,-4/3,3,-4,25/12],[0,0,0,0,1])

#以下是椭圆方程
def three_diff(b,c,f,t0,h,n):
    #初始化f
    tv=[t0+i*h for i in range(1,n)]
    fv=[f(tv[i]) for i in range(n-1)]
    fv=np.array(fv).astype(np.float64)
    #初始化A
    a1=[2 for i in range(n-1)]
    a2=[-1 for i in range(n-2)]
    A=1/h/h*(np.diag(a2,-1)+np.diag(a1)+np.diag(a2,1))
    A=A.astype(np.float64)
    #初始化B
    b1=[1 for i in range(n-2)]
    B=b/2/h*(np.diag(b1,1)-np.diag(b1,-1))
    B=B.astype(np.float64)
    #初始化C
    c1=[c(tv[i]) for i in range(n-1)]
    C=np.diag(c1) 
    C=C.astype(np.float64)
    return tv,fv,A,B,C

def dirichlet(b,c,f,t0,h,n,u0,un):
    tv,fv,A,B,C=three_diff(b,c,f,t0,h,n)
    fv[0]=fv[0]+u0/h/h+b/2/h*u0
    fv[n-2]=fv[n-2]+un/h/h-b*un/2/h
    return tv,solve(A+B+C,fv)

#对dirichlet边界问题画图，b是系数，c是系数函数，f是右端函数，t0是起始点，h是步长列表，n是迭代轮数列表，
#u0,un是边界条件，s是精确解函数，返回双对数收敛阶图像数据
def show_conv_dirichlet(b,c,f,t0,h,n,u0,un,s):
    x=[]
    y=[]
    for i in range(len(h)):
        print('h=',h[i])
        t, u=dirichlet(b,c,f,t0,h[i],n[i],u0,un)
        y.append(np.log(abs(u[int(n[i]/3)]-s(t[int(n[i]/3)]))/abs(t[int(n[i]/3)])))
        x.append(np.log(h[i]))
    return x,y

#robin边界问题，u(0)=u0, u'(1)+alpha*u(1)=g
def robin(b,c,f,t0,h,n,u0,alpha,g):
    beta=(3*h+2*alpha*h*h)/(h*b-2)
    tv,fv,A,B,C=three_diff(b,c,f,t0,h,n)
    fv[0]=fv[0]+u0/h/h+b/2/h*u0
    fv[n-2]=g-fv[n-2]*beta
    C[n-2,n-2]=-c(tv[n-2])*beta
    B[n-2,n-3]=1/2/h
    B[n-2,n-2]=-2/h
    A[n-2,n-3]=beta*(1/h/h+b/2/h)
    A[n-2,n-2]=-2*beta/h/h
    return tv,solve(A+B+C,fv)

#对robin边界问题画图，b是系数，c是系数函数，f是右端函数，t0是起始点，h是步长列表，n是迭代轮数列表，
#u0,alpha,g是边界条件，s是精确解函数，返回双对数收敛阶图像数据
def show_conv_robin(b,c,f,t0,h,n,u0,alpha,g,s):
    x=[]
    y=[]
    for i in range(len(h)):
        print('h=',h[i])
        t, u=robin(b,c,f,t0,h[i],n[i],u0,alpha,g)
        y.append(np.log(abs((u[-1]-s(t[-1]))/s(t[-1]))))
        x.append(np.log(h[i]))
    return x,y
