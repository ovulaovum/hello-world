function label_extend = softmax_labext(label)
% 计算标签示性矩阵
% 将每个标签扩展为一个k维横向量（k为标签类别数），若样本i属于第j类，则
% label_extend（i，j）= 1，否则label_extend（i，j）= 0。
label_extend = [full(sparse(label,1:length(label),1))]';