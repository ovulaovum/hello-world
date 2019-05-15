function vk = softmax_vk(X,ik,label_extend,theta1,theta,v,m)
% 本函数计算Prox—SVRG算法里的vk
P = softmax_ProbMat_irow(ik,X,m,label_extend,theta);
P1 = softmax_ProbMat_irow(ik,X,m,label_extend,theta1);
vk=v+softmax_comp_grad(ik,X,label_extend,P1)-softmax_comp_grad(ik,X,label_extend,P);