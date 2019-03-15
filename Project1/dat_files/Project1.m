% PSC Project 1
% Dustin (Ting-Hsuan) Ma
clear all; close all; clc;

% import data

data1 = importdata('./2d_membrane_initial.dat');
data2 = importdata('./2d_membrane_final.dat');
dataExact = importdata('./2d_membrane_exact.dat');

%% Graphing Matrix

NX = 41;
NY = 21;

init = zeros(NY,NX);
final = zeros(NY,NX);
exact = zeros(NY,NX);

for i = 1:NY
    init(i,:)=data1(i,:);
    final(i,:)=data2(i,:);
    exact(i,:)=data2(i,:);
end
x = linspace(0,4,NX);
y = linspace(0,2,NY);
[x,y] = meshgrid(x,y);

figure();
surf(x,y,init)
colorbar;
title('Initial')
xlabel('X')
ylabel('Y')
zlabel('Amplitude')

figure();
surf(x,y,final)
colorbar;
title('Final')
xlabel('X')
ylabel('Y')
zlabel('Amplitude')

figure();
surf(x,y,exact)
colorbar;
title('Exact')
xlabel('X')
ylabel('Y')
zlabel('Amplitude')