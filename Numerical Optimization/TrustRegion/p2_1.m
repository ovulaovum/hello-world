d=linspace(0,2,101);%deltaȡ0��2
pr(0,-1);%����(0,-1)�����ĵȸ���
[~,g,G] = m(0,-1);
s1=zeros(2,101);
for i=1:101
[s1(:,i),~,~,~,~]=trust(g,G,d(i));%����x^*
end
figure;
plot(s1(1,:),s1(2,:));%x^*��ͼ��
%���滻��(0,0.5)��ͼ
figure;
pr(0,0.5);
[~,g,G] = m(0,0.5);
s2=zeros(2,101);
for i=1:101
[s2(:,i),~,~,~,~]=trust(g,G,d(i));
end
figure;
plot(s2(1,:),s2(2,:));
function []=pr(a,b)%�ú������ڻ����ȸ���
x = linspace(a-2,a+2,201);
y = linspace(b-2,b+2,201);
[X,Y] = meshgrid(x,y);
[f,g,G] = m(a,b);
Z = f+g(1).*X+g(2).*Y+1/2*(G(1,1).*X.^2+G(2,2).*Y.^2+2*G(1,2).*X.*Y);
contour(X,Y,Z);
end