clc
clearvars
close all

%% Initialising
py_dir = "C:\Users\vvh19\OneDrive\Documents\GitHub\SatRelMDPC\src\python";
py_script_rel_dir = "HelloWorld";
py_script_rel_dir = "HW";
conda_env_name = "base";

%% Setup Command Prompt Command
curr_dir = string(pwd);
command = "@echo off && cd " + py_dir + " && conda activate " + conda_env_name + " && python " + py_script_rel_dir + ".py && cd " + curr_dir + " && exit";
command

%% Run Python Through Comand Prompt
[status, cmdout] = system(command);
cmdout