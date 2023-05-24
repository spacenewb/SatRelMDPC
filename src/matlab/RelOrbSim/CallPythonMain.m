clc
clearvars

%% Initialising
py_dir = "C:\Users\vvh19\OneDrive\Documents\GitHub\SatRelMDPC\src\python";
conda_env_name = "base";
py_script_name = "Linearise";
py_func_name = "helloworld";
arrayd = [1,4,3,6,8,2];
arraye = [6,7,8,9,3,4];

%% Setup Command Prompt Command
curr_dir = string(pwd);
command = "@echo off && cd " + py_dir + " && conda activate " + conda_env_name + " && python " + py_script_name + ".py " + '"[' + num2str(arrayd) + ']"' + " " + '"[' + num2str(arraye) + ']"' + " && cd " + curr_dir + " && exit";
command

%% Run Python Through Comand Prompt
[status, cmdout] = system(command)
