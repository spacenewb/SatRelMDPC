clc
clearvars
close all
delete('Export\*.csv')

% Uncomment below if ModelDiscoveryAccuracyTestCampaign.csv results need to be cleared
% fclose(fopen('ModelDiscoveryAccuracyTestCampaign.csv', 'w')); 

tic
g0 = 9.81;
MU = 3.98600433e+5; % From DE405

%% Model Initialisation
model_name = "RelOrbSim";
timestep_lims = [0.01, 0.01]; % [min, max]
J2_flag = 0;
% Y0_Target = [7106.14, 0.05, deg2rad(98.3), deg2rad(0), deg2rad(270), deg2rad(0)]';
% Y0_Chaser = [7106.14, 0.0501, deg2rad(98.31), deg2rad(0), deg2rad(270), deg2rad(0)]';
Y0_Target = [7106.14, 0.5, deg2rad(98.3), deg2rad(0), deg2rad(270), deg2rad(0)]';
Y0_Chaser = [7106.14, 0.501, deg2rad(98.31), deg2rad(0), deg2rad(270), deg2rad(0)]';
T_orb = 2*pi*sqrt(max([Y0_Target(1), Y0_Chaser(1)])^3/MU);
M0 = 12; 
Isp = 450;
Mp = 0.5;
N_orbs = 600*(1/T_orb); % Remember that linearisation interval in python is half of this time duration due to train and validation sets
write_flag = 1;
ctrl_flag = 1;

%% Run Model
disp("Simulation Initialising!")
T_sim = N_orbs*max(2*pi*sqrt([Y0_Target(1)^3, Y0_Chaser(1)^3]/MU));
load_system(model_name)
if diff(timestep_lims) == 0
    set_param(model_name, 'Solver', 'ode3', 'FixedStep', num2str(timestep_lims(1)))
else
    set_param(model_name, 'Solver', 'ode113', 'MinStep', num2str(timestep_lims(1)), 'MaxStep', num2str(timestep_lims(2)))
end
set_param(model_name, 'StopTime', num2str(T_sim))
w = warning('off','all');
Sim = sim((model_name + '.slx'), 'CaptureErrors','on');
warning(w);
disp("Simulation Completed Successfully!")
disp(Sim.ErrorMessage)

%% Read Outputs
SimData = load(model_name + "_Out.mat").Data;
times = SimData(1,:)';
a_f_RTH = SimData(2:4,:)';
rho = SimData(5:7,:)';
drho = SimData(8:10,:)';
ddrho = SimData(11:13,:)';
kep_T = SimData(14:19,:)';
kep_C = SimData(20:25,:)';
params_T = SimData(26:28,:)';
params_C = SimData(29:31,:)';
disp("Simulation Data Read Sucessfully!")

%% Write Outputs
if write_flag == 1
    writematrix(times,'Export/times.csv','WriteMode','overwrite','FileType','text');
    writematrix(a_f_RTH,'Export/a_f_RTH.csv','WriteMode','overwrite','FileType','text');
    writematrix(rho,'Export/rho.csv','WriteMode','overwrite','FileType','text');
    writematrix(drho,'Export/rho_dot.csv','WriteMode','overwrite','FileType','text');
    writematrix(ddrho,'Export/rho_dotdot.csv','WriteMode','overwrite','FileType','text');
    writematrix(kep_T,'Export/kep_T.csv','WriteMode','overwrite','FileType','text');
    writematrix(kep_C,'Export/kep_C.csv','WriteMode','overwrite','FileType','text');
    writematrix(params_T,'Export/params_T.csv','WriteMode','overwrite','FileType','text');
    writematrix(params_C,'Export/params_C.csv','WriteMode','overwrite','FileType','text');
    disp("Simulation Data Exported Successfully!");
else
    disp("Simulation Data Not Exported!")
end

%% Run Python Script for Model Identification
%% Initialising
disp("Initialisaing Python Environment!")
py_dir = "C:\Users\vvh19\OneDrive\Documents\GitHub\SatRelMDPC\src\python";
conda_env_name = "base";
py_script_name = "MAIN";
py_func_name = "EstimateSindyWithControl_MSE";

%% Setup Command Prompt Command
curr_dir = string(pwd);
command = "@echo off && cd " + py_dir + " && conda activate " + conda_env_name + " && python -c " + '"from ' + py_script_name + " import *; " + py_func_name + "()" + '"' + " && cd " + curr_dir + " && exit";

%% Run Python Through Comand Prompt
disp("Running Python Function Via Command Through Conda Environment in Command Prompt!")
[status, cmdout] = system(command);

%% Parsing cmdout into matlab variable CMDOUT 
disp("Parsing Python Output Into Matlab Array and Exporting To CSV file '-append'!")
CMD_OUT_labels = ['Ecc', 'Theta0', 'Ts', 'LinInt', 'MSE_Train', 'MSE_Test', 'CPU Ex. Time'];
CMD_OUT = str2num(cmdout)';
ExecTime = toc;

%% Setting Output and Exporting CSV via '-append'
MAIN_OUT_labels = [CMD_OUT_labels, 'Ex. Time Tot.']
MAIN_OUT = [CMD_OUT, ExecTime]
dlmwrite('ModelDiscoveryAccuracyTestCampaign.csv',MAIN_OUT,'delimiter',',','-append');
