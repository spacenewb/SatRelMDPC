clc
clearvars
close all

Controller_Engaged_flag = 1; % 0--> No Controller; 1--> Controller Engaged
NumLinCycles = 10; % Num. of Lin. cycles to simulate
LinInt = 300; % Linearisation Interval

py_script_name = "linearise";
delete_temp_flag = 1;

%% Model Initialisation
g0 = 9.81;
MU = 3.98600433e+5; % From DE405

model_name = "RelOrbSim";
timestep = 0.1; % Measurement Sampling Time & also Simulation Step Time
J2_flag = 1;
Isp = 250;

Ts_MPC = 5; % MPC Prediction Sample Time
pHorizon = 60; % MPC Prediction Horizon - Samples
cHorizon = 12; % MPC Control Horizon - Samples
RangeLims = [13.5 10]; % Range Limits for MPC

CostA = 10; % Min. Lim Violation Exponent
CostB = 100; % Max Lim Violation Slope
CostSigma = 3; % Sigma Accuracy of PDF 
CostRelG = 1e3; % Relative gain (Inputs vs. TrajError)

Tmax = 0.001; % [kN]

Y0_Target = [7106.14, 0.5, deg2rad(98.3), deg2rad(0), deg2rad(270), deg2rad(0)]';
Y0_Chaser = [7106.14, 0.501, deg2rad(98.31), deg2rad(0), deg2rad(270), deg2rad(0)]';
M0 = 12;

%% Loop to run through subsequent Linearisation Cycles
ctrl_flag = 0;
LinSysMat = 0;
T0_next = 0;
LinSysMat = zeros(6,6);
disp("Simulation Initialising!")
for LinCycleIdx = 1:NumLinCycles
    if LinCycleIdx > 1
        ctrl_flag = 1;  
        Y0_Target = Y0_Target_next; 
        Y0_Chaser = Y0_Chaser_next;
        M0 = M0_next;
    end
    % Run Model
    disp("Linearisation Cycle " + num2str(LinCycleIdx))
    load_system(model_name)
    set_param(model_name, 'Solver', 'ode3', 'FixedStep', num2str(timestep))
    set_param(model_name, 'StopTime', num2str(LinInt))
    w = warning('off','all');
    Sim = sim((model_name + '.slx'), 'CaptureErrors','on');
    curr_sim_time = get_param(model_name, 'SimulationTime');
    warning(w);
    %disp(Sim.ErrorMessage)
    clc;
    
    % Save Simulation Data
    SimData = [Sim.tout + T0_next*(LinCycleIdx-1), Sim.Data];
    save('temp/SimData', 'SimData')

    % Set Final State - Used for Initial State For Next Sim
    Y0_Target_next = Sim.Data(end,13:18)';
    Y0_Chaser_next = Sim.Data(end,19:24)';
    M0_next = Sim.Data(end,31);
    T0_next = Sim.tout(end);
    
    % Setup Command Prompt Command
    command = "@echo off && cd " + pwd() + " && conda activate base && python " + py_script_name + ".py " + " && exit";
    [status, cmdout] = system(command); % Run Python Through Comand Prompt
    load('temp/LinSysMat.mat')

    % Save All Data About Current LinCycle
    File1Name = "temp/cycles/SimData_LinCycle_" + num2str(LinCycleIdx);
    save(File1Name, 'SimData');
    File2Name = "temp/cycles/LinSysMat_LinCycle_" + num2str(LinCycleIdx);
    save(File2Name, 'LinSysMat');
end
disp("Simulation Completed Successfully!")

%% Combine all Cycles Data

Global_SimData = [];
Global_LinSysMat = [];
for LinCycleIdx = 1:NumLinCycles

    % Load Data About All LinCycles
    File1Name = "temp/cycles/SimData_LinCycle_" + num2str(LinCycleIdx);
    File2Name = "temp/cycles/LinSysMat_LinCycle_" + num2str(LinCycleIdx);

    Curr_SimData = load(File1Name).SimData;
    Curr_LinSysMat = load(File2Name).LinSysMat;

    % Combine into one MAT file
    Global_SimData = [Global_SimData; Curr_SimData];
    Global_LinSysMat = [Global_LinSysMat; Curr_LinSysMat];

    save('temp/GlobalSimData', 'Global_SimData');
    save('temp/GlobalLinSysMat', 'Global_LinSysMat');
end

%% Test Global Data Authenticity - Plot Mass - Should not chnage during cycle 1
load('temp/GlobalSimData.mat')
Ranges = vecnorm(Global_SimData(:,5:7)');
Thrusts = Global_SimData(:,33);
times = Global_SimData(:,1);

figure();

subplot(2,1,1)
plot(times, Ranges, 'k','LineWidth', 2)
xline(LinInt, 'b--', 'LineWidth', 2)
yline(RangeLims(1), 'r:', 'LineWidth', 2)
yline(RangeLims(2), 'r:', 'LineWidth', 2)
xlabel('Time [s]', 'interpreter', 'latex')
ylabel('Rel. Range [km]', 'interpreter', 'latex')
title('Rel. Range', 'interpreter', 'latex')
set(gca,'TickLabelInterpreter','latex')
grid minor

subplot(2,1,2)
plot(times, Thrusts*1e3, 'g','LineWidth', 2)
xline(LinInt, 'b--', 'LineWidth', 2)
xlabel('Time [s]', 'interpreter', 'latex')
ylabel('Thrust [N]', 'interpreter', 'latex')
title({"","",'Thrust'}, 'interpreter', 'latex')
set(gca,'TickLabelInterpreter','latex')
grid minor

if Controller_Engaged_flag == 0
    sgt = sgtitle("MPC performance: Controller Disengaged");
else
    sgt = sgtitle("MPC performance: Controller Engaged");
end

sgt.Interpreter = 'latex';
sgt.FontSize = 12;
txt1 = "$T_s: (" + Ts_MPC + ",~" + timestep + ") [s];~"...
    + "H: (" + pHorizon + ",~" + cHorizon + ") [steps];~"...
    + "Bound: (" + RangeLims(2) + " - " + RangeLims(1) + ") [km];~"...
    + "T_{lin}: " + LinInt + " [s];~" + "Cycles: " + NumLinCycles + "$";
txt2 = "$CostA: " + CostA + ";~CostB: " + CostB...
    + ";~CostSigma: " + CostSigma + ";~CostRelG: " + CostRelG + "$";
annotation('textbox', [0.5, 0.92, 0, 0], 'string', txt1, 'FitBoxToText','on',...
    'interpreter','latex', 'HorizontalAlignment','center',...
    'VerticalAlignment','middle','LineStyle','none','Color','red')
annotation('textbox', [0.5, 0.885, 0, 0], 'string', txt2, 'FitBoxToText','on',...
    'interpreter','latex', 'HorizontalAlignment','center',...
    'VerticalAlignment','middle','LineStyle','none','Color','red')

%% Delete All Temp Files in temp Directory
if delete_temp_flag == 1
    delete('temp/LinSysMat.mat');
    delete('temp/SimData.mat');
    delete('temp/cycles/*.mat');
end
