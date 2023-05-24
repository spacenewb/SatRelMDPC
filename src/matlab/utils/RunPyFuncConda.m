clc
clearvars
close all

%% Initialising
py_dir = "C:\Users\vvh19\OneDrive\Documents\GitHub\SatRelMDPC\src\python";
py_script_name = "HW";
py_func_name = "Hd";
conda_env_name = "base";

py_script_name = "MAIN";
py_func_name = "EstimateSindyWithControl_MSE";

%% Setup Command Prompt Command
curr_dir = string(pwd);
command = "@echo off && cd " + py_dir + " && conda activate " + conda_env_name + " && python -c " + '"from ' + py_script_name + " import *; " + py_func_name + "()" + '"' + " && cd " + curr_dir + " && exit";

%% Run Python Through Comand Prompt
[status, cmdout] = system(command);

%% Parsing cmdout into matlab variable CMDOUT
CMDOUT = str2num(cmdout)