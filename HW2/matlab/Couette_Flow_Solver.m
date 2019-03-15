%% ME2054
% Homework 2 - Couette Flow Solver
clear all; close all; clc;
%% RE = 10 PG = 0, 2, -2
figure
for l = 1:3
    Re = [10 10 10];
    PG = [0 2 -2];
    color = ['r','b','k'];
    
    % initializing variables
    mu = 1;
    rho = 1;
    H = 2;
    numInt = 20.0;
    
    % calculating variable depended values
    dy = H/(numInt-1);
    nu = mu/rho;
    Uplate = Re(l)*nu/H;
    dp = PG(l)*(-rho);
    dt = 0.5*dy^2/nu;
    steadyT = H^2/nu;
    nTimeStep = steadyT/dt;
    y = linspace(0,H,numInt);
    
    % allocating vector memory
    v = zeros(1,20);
    w = zeros(1,20);
    u = zeros(1,20);
    
    % Calculating exact solution
    for i = 1:numInt
        u(i) = Uplate * (y(i)/H) + dp/(2*mu) * (y(i)^2 - H * y(i));
        u(1) = 0;
        u(end) = Uplate;
    end
    
    
    % Calculating numerical solution
    for i = 1:nTimeStep
        for j = 2:numInt-1
            v(j) = dt*(PG(l)+mu*(w(j+1)-2*w(j)+w(j-1))/dy^2)+w(j);
            w(j) = v(j);
            w(1) = 0;
            w(end) = Uplate;
        end
        
    end
    
    % plotting
    numerical = w/Uplate;
    exact = u/Uplate;
    z = y/H;
    
    plot(numerical,z,color(l))
    hold on
    plot(exact,z,color(l))
end

title('Renolds Number = 10, PG = 0, 2, -2')
xlabel('u/Vplate')
ylabel('y/H')
ylim([0,1])
legend('PGnu = 0','PGex = 0','PGnu = 2','PGex = 2','PGnu = -2','PGex = -2','Location','northwest')
hold off
%% RE = 1 5 10 PG = 2
figure
for l = 1:3
    Re = [1 5 10];
    PG = [2 2 2];
    color = ['r','b','k'];
    
    % Re = 10; PG = 2;
    % initializing variables
    mu = 1;
    rho = 1;
    H = 2;
    numInt = 20.0;
    
    % calculating variable depended values
    dy = H/(numInt-1);
    nu = mu/rho;
    Uplate = Re(l)*nu/H;
    dp = PG(l)*(-rho);
    dt = 0.5*dy^2/nu;
    steadyT = H^2/nu;
    nTimeStep = steadyT/dt;
    y = linspace(0,H,numInt);
    
    % allocating vector memory
    v = zeros(1,20);
    w = zeros(1,20);
    u = zeros(1,20);
    
    % Calculating exact solution
    for i = 1:numInt
        u(i) = Uplate * (y(i)/H) + dp/(2*mu) * (y(i)^2 - H * y(i));
        u(1) = 0;
        u(end) = Uplate;
    end
    
    
    % Calculating numerical solution
    for i = 1:nTimeStep
        for j = 2:numInt-1
            v(j) = dt*(PG(l)+mu*(w(j+1)-2*w(j)+w(j-1))/dy^2)+w(j);
            w(j) = v(j);
            w(1) = 0;
            w(end) = Uplate;
        end
        
    end
    
    % plotting
    numerical = w/Uplate;
    exact = u/Uplate;
    z = y/H;
    
    plot(numerical,z,color(l))
    hold on
    plot(exact,z,color(l))
end
title('Renolds Number = 1, 5, 10, PG = 2')
xlabel('u/Vplate')
ylabel('y/H')
ylim([0,1])
legend('REnu = 1','REex = 1','REnu = 5','REex = 5','REnu = 10','REex = 10','Location','northwest')
hold off