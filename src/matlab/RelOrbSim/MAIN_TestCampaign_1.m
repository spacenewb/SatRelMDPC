clc
clearvars
close all

fclose(fopen('ModelDiscoveryAccuracyTestCampaign.csv', 'w')); % to open and close - erase data - in the test campaign csv output file

%% MAIN LOOP TRIAL

SMA = 7106.14; % KM
Del_ECC = 0.001; % Chaser = Target + Del
INC = 98; % Deg
Del_INC = 0.01; % [Deg] Chaser = Target + Del
RAAN = 0; % Deg
OMEGA = 270; % Deg

Dat_File_Content = {'SMA', 'Del_ECC', 'INC', 'Del_INC', 'RAAN', 'OMEGA';
                    '[KM]',  '[-]',  '[Deg]','[Deg]',   '[Deg]','[Deg]';
                    SMA,    Del_ECC,   INC,   Del_INC,   RAAN,   OMEGA};
writecell(Dat_File_Content, 'ModelDiscoveryAccuracyTestCampaign.dat', 'Delimiter', '\t', 'FileType','text');

%%

% Setup Control Variable Matrix
CV_num_discretisations = 5; % Num of samples for each Control Variable - eg. How many eccentricities we test for?

num_CV = 4;
SampFeq_lims = [1, 0.01];
LinInt_lims = [300, 10];
Ecc_lims = [0.6, 0];
Theta0_lims = [pi, 0];

Tot_num_tests = CV_num_discretisations^num_CV;

SampFeq_vec = linspace(SampFeq_lims(2), SampFeq_lims(1), CV_num_discretisations);
LinInt_vec = linspace(LinInt_lims(2), LinInt_lims(1), CV_num_discretisations);
Ecc_vec = linspace(Ecc_lims(2), Ecc_lims(1), CV_num_discretisations);
Theta0_vec = linspace(Theta0_lims(2), Theta0_lims(1), CV_num_discretisations);

%% Run simulations
% Uncomment below if ModelDiscoveryAccuracyTestCampaign.csv results need to be cleared
fclose(fopen('ModelDiscoveryAccuracyTestCampaign.csv', 'w'));
Test_idx = 1;
Test_Time_Tot = 0;

WBar = waitbar(0,"Running System Identification Accuracy Test Campaign...", "Name","Progress Bar: Test Campaign");

for II1 = 1:size(SampFeq_vec,2)
    for II2 = 1:size(LinInt_vec,2)
        for II3 = 1:size(Ecc_vec,2)
            for II4 = 1:size(Theta0_vec,2)
    
                delete('Export\*.csv')

                tic
            
                CV = [SampFeq_vec(II1), LinInt_vec(II2), Ecc_vec(II3), Theta0_vec(II4)];

                disp("Test Num: " + string(Test_idx))
                disp("Control Variables: ")
                disp(string(CV))
                
                timestep = CV(1); %
                LinInt  = CV(2);
                ECC = CV(3);
                THETA0 = CV(4);
                
                g0 = 9.81;
                MU = 3.98600433e+5; % From DE405
                
                %% Model Initialisation
                model_name = "RelOrbSim";
                
                J2_flag = 0;
                Y0_Target = [SMA, ECC, deg2rad(INC), deg2rad(RAAN), deg2rad(OMEGA), THETA0]';
                Y0_Chaser = [SMA, (ECC+Del_ECC), deg2rad(INC+Del_INC), eg2rad(RAAN), deg2rad(OMEGA), THETA0]';
                T_orb = 2*pi*sqrt(max([Y0_Target(1), Y0_Chaser(1)])^3/MU);
                M0 = 12; 
                Isp = 450;
                Mp = 0.5;
                N_orbs = (LinInt*2)*(1/T_orb); % Remember that linearisation interval in python is half of this time duration due to train and validation sets
                write_flag = 1;
                ctrl_flag = 1;
                
                %% Run Model
                disp("Simulation Initialising!")
                T_sim = N_orbs*max(2*pi*sqrt([Y0_Target(1)^3, Y0_Chaser(1)^3]/MU));
                load_system(model_name)
                
                set_param(model_name, 'Solver', 'ode3', 'FixedStep', num2str(timestep))
                
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

                CMD_OUT = [ECC, THETA0, timestep, LinInt, CMD_OUT(5), CMD_OUT(6), CMD_OUT(7)]; % To take CV values from Matlab Instead of Python Code Output

                ExecTime = toc;
                
                %% Setting Output and Exporting CSV via '-append'
                format short
                MAIN_OUT_labels = [CMD_OUT_labels, 'Ex. Time Tot.'];
                MAIN_OUT = [CMD_OUT, ExecTime]
                format shortE
                dlmwrite('ModelDiscoveryAccuracyTestCampaign.csv',MAIN_OUT,'delimiter',',','-append');
                
                
                Test_Time_Tot = Test_Time_Tot + ExecTime;
                ETA = (((Test_Time_Tot/Test_idx)*Tot_num_tests) - Test_Time_Tot)/60; % In Minutes

                Test_idx = Test_idx + 1;
                Loadbar_text = "(" + num2str(Test_idx) + "/" + num2str(Tot_num_tests) + ") " + "Estimated Time Left: " + num2str(ETA) + " mins";
                waitbar(Test_idx/Tot_num_tests,WBar,["Running System Identification Accuracy Test Campaign..." + newline + Loadbar_text]);
            end
        end
    end
end

close(WBar)