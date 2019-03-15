clear all; close all; clc;
endtimes = importdata('./endtime_4times.csv');

N = 12;
idx = 59;
LY = 2;
x = linspace (0,LY,idx);

figure(1);
hold on
plot(x,endtimes.data(:,1));
plot(x,endtimes.data(:,2));
plot(x,endtimes.data(:,3));
title('Time = 1 [s]')
hold off

figure(2);
hold on
plot(x,endtimes.data(:,4));
plot(x,endtimes.data(:,5));
plot(x,endtimes.data(:,6));
title('Time = 5 [s]')
hold off

figure(3);
hold on
plot(x,endtimes.data(:,7));
plot(x,endtimes.data(:,8));
plot(x,endtimes.data(:,9));
title('Time = 10 [s]')
hold off

figure(4);
hold on
plot(x,endtimes.data(:,10));
plot(x,endtimes.data(:,11));
plot(x,endtimes.data(:,12));
title('Time = 12 [s]')
hold off