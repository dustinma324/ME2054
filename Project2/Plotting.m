clear all; close all; clc;

dat = importdata('./TimeVsMesh.csv');

mesh = dat.data(:,2);
CPU = dat.data(:,3)*1E-6;
DDOT = dat.data(:,4)*1E-6;
DAXPY = dat.data(:,5)*1E-6;
multKernel = dat.data(:,6)*1E-3;
cublasDdot = dat.data(:,7)*1E-3;
cublasDaxpy = dat.data(:,8)*1E-3;

hold on
plot(mesh,CPU,'LineWidth',3);
plot(mesh,DDOT,'LineWidth',3);
plot(mesh,DAXPY,'LineWidth',3);
plot(mesh,multKernel,'LineWidth',3)
plot(mesh,cublasDdot,'LineWidth',3)
plot(mesh,cublasDaxpy,'LineWidth',3)
hold off

title('Time Vs. Mesh')
xlabel('Mesh Size (2^i, i=1,2,3...12)')
legend('CPU','Ddot','Daxpy','MultKernel','CublasDdot','CublasDaxpy','Location','northwest')
grid on