clear all; close all; clc;

exact = importdata('./exact.txt');
CPU = importdata('./CPU.txt');
GPU = importdata('./GPU.txt');

y = linspace (0,2,21);

idx = [11 21 31];

quarter(:,1) = exact(:,31);
quarter(:,2) = CPU(:,31);
quarter(:,3) = GPU(:,31);

figure();
hold on
for i = 1:3
plot(y,quarter(:,i));
end
legend('Exact','CPU','GPU')
title('Time 7[s], 0.5LX')
xlabel('width, Ly = 2')
ylabel('Amplitude')
hold off