# _*_ coding:utf-8 _*_
import numpy as np
import sympy as sp
from sympy import cos, sin
import matplotlib.pyplot as plt
sp.init_printing(use_unicode=False, wrap_line=False, no_global=True)
x=sp.symbols('x')
Pi=np.pi
def FFT(f,xx):
    n=len(xx)
    p=int(np.log2(n))
    a1=[]
    for i in range(n):
        a1.append(complex(f.subs(x,xx[i])))
    a2=[0+0j]*n
    w=[]
    for m in range(int(n/2)):
        w.append(np.exp(-1j*2*Pi*m/n))
    for q in range(1,p+1):
        if q/2-int(q/2)<0.4:
            for i in range(2**(q-1)):
                for k in range(2**(p-q)):
                    a1[k*2**q+i]=a2[k*2**(q-1)+i]+a2[k*2**(q-1)+i+2**(p-1)]
                    a1[k*2**q+i+2**(q-1)]=(a2[k*2**(q-1)+i]-a2[k*2**(q-1)+i+2**(p-1)])*w[k*2**(q-1)]
        else:
            for i in range(2**(q-1)):
                for k in range(2**(p-q)):
                    a2[k*2**q+i]=a1[k*2**(q-1)+i]+a1[k*2**(q-1)+i+2**(p-1)]
                    a2[k*2**q+i+2**(q-1)]=(a1[k*2**(q-1)+i]-a1[k*2**(q-1)+i+2**(p-1)])*w[k*2**(q-1)]
    if p/2-int(p/2)<0.4:
        a2=a1    
    return a2

xx=np.linspace(-Pi,7*Pi/8,16)
f=x**2*cos(x)
c=FFT(f,xx)

for i in range(16):
    c[i]=c[i]*((-1)**i)/8
a=np.real(c)
b=np.imag(c) 

s=0
for k in range(16):
    s=s+a[k]*cos(k*x)+b[k]*sin(k*x)

xxx=np.linspace(-Pi,Pi,1000)
y1=[]
y2=[]
for i in range(1000):
    y1.append(s.subs(x,xxx[i]))
    y2.append(f.subs(x,xxx[i]))
plt.figure()
plt.plot(xxx,y1)
plt.plot(xxx,y2)
plt.show()