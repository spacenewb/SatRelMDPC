clc
clearvars

%% testing python reading mat files
% save 2d arrays in matfile
rng('default')
A = rand(10000,21); 
save('PyLinIPVars', 'A');

%% Initialising
py_dir = "C:\Users\vvh19\OneDrive\Documents\GitHub\SatRelMDPC\src\python";
conda_env_name = "base";
py_script_name = "Linearise";

%% Setup Command Prompt Command
curr_dir = string(pwd);
command = "@echo off && cd " + py_dir + " && conda activate " + conda_env_name + " && python " + py_script_name + ".py " + " && cd " + curr_dir + " && exit";

%% Run Python Through Comand Prompt
[status, cmdout] = system(command);
load('LinSysMat.mat')
