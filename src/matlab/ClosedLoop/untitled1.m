clc
clearvars

X = 0:0.0001:1;

ranges = rand(1000,1);

lims = [0.3 0.8];

gains = [1 1];

costs = @(ranges) (lims(1) - min(lims(1), ranges))*gains(1) + (max(lims(2), ranges) - lims(2))*gains(2);

sum(costs(ranges))/length(costs(ranges))

figure()
plot(X, costs(X))