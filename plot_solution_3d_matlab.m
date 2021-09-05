A=importdata('test.dat');
x = A(:,1);
y = A(:,2);
z = A(:,3);
true = A(:,4);
pred = A(:,5);
markerSize = 500;

scatter3(x,y,z,markerSize,true,'filled')
colorbar
axis equal;
scatter3(x,y,z,markerSize,pred,'filled')
colorbar
markerSize = 100;
save('test_change_rhs.dat','A');

[X,Y,Z] = meshgrid(x,y,z);
Pred = meshgrid(pred,pred,pred);
figure; 
surf(X,Y,Z,Pred)
colorbar
%# make sure it will look like a sphere
