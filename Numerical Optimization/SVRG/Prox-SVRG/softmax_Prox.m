function thetak = softmax_Prox(thetak_1,yita_k,beta1,epsil,lambda,vk)
% 本函数计算Prox函数的值，本次编写为了简便，采用了梯度下降法。
% thetak_1是上一步迭代的结果
% yita_k是学习率，我们这里为了简便起见设置成常数。
% eps是梯度的精度
% beta1是这里采用的GD的步长
thetak = thetak_1;
p = yita_k*vk+lambda*thetak; %初始化梯度向量
while(norm(p)>epsil)
    thetak=thetak-beta1*p;
    p = thetak-thetak_1+yita_k*vk+lambda*thetak;
end