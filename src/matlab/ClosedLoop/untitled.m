clc
clearvars
close all

Sys_A = [  0	                    0	                    0	                    1	                    0	                0;
            0	                    0	                    0	                    0	                    1	                0;
            0	                    0	                    0	                    0	                    0	                1;
            2.22137111013988e-05	7.48690350981646e-07	0	                    0	                    0.00634559620392871	0;
            -7.48690350981644e-07	-1.23311366355553e-05	0	                    -0.00634559620392871	0	                0;
            0	                    0	                    -1.15347137817196e-05	0	                    0	                0];

X0 = [-7.10619400811799	-9.92113036647141e-21	-0.618887441995133	-3.63428453405587e-23	0.0173193267761480	-4.16458329597728e-19];

%% Initialisation
a = 1;      b = 1;
Mass = 12;
Sys_B = [zeros(3,3); eye(3)./Mass];

% Horizons
Ts = 1;       pH = 300;       cH = 30;

%% Thrust Profile
U = @(t) (a*exp(-b*t)).*(t<=Ts*cH);
thrust_pointing = @(X) -X(1:3)./norm(X(1:3));

%% Propagation
X_dot = @(t, X) Sys_A*X + Sys_B*thrust_pointing(X)*U(t);
tspan = 0:Ts:Ts*pH;
ode_opt = odeset('RelTol',1e-5,'AbsTol',1e-6);
[~,y] = ode45(@(t, X) X_dot(t, X), tspan, X0, ode_opt);
ranges = vecnorm(y(:,1:3)');

%% Plot Prediction
figure();
plot(tspan, ranges)

%% Cost Calculations
% Traj Limits
m_rng = 1.43;% 12; % minimum range bounds [km]
M_rng = 1.425;% 8; % maximum range bounds [km]

% Cost
a_cost = 10; % Amplitude factor for Area under minimum violation cost
b_cost = 100; %  Amplitude factor/ slope for maximum violation cost
sigma_width = 3; % Determines the width of bell curve's sigma limit
CtrlCost_Gain = 1e3;

% CostFn accepts Ranges in meters only
costFn = @(Ranges) (Ranges - M_rng) ./ ( 1./b_cost + exp(M_rng - Ranges) )...
    - (Ranges + M_rng) ./ ( 1./b_cost + exp(M_rng + Ranges) )...
    + ( (a_cost.^4) ./ ((m_rng./sigma_width).*sqrt(2.*pi)) ) .* exp(-Ranges.^2 ./ (2.*(m_rng./sigma_width).^2));

TrajCost = sum(costFn(ranges));
CtrlCost = sum(U(tspan))*CtrlCost_Gain/(cH*Ts)
TotalCost = TrajCost + CtrlCost

figure();
L=0:0.01:5;
plot(L, costFn(L))

figure();
L=0:Ts:cH*Ts;
plot(L,U(L))