clc
clearvars
close all

%% Plot the results of the Prediction Accuracy test Campaign
% view_type ==> '3d'-->3d isometric; 'XY'-->(X-Y), 'YZ'-->(Y-Z), 'ZX'-->(Z-X)
view_type = '3d'; 

% Which MSE to plot as z axis (MSE_test, MSE_train, Py_time)
z_var = 'Py_time';

% Which Test Campaign Data to plot
TestCampaign_ID = 1;

%%
% Test Data CSV Directory:
Data_CSV_dir = "C:\Users\vvh19\OneDrive\Documents\GitHub\SatRelMDPC\src\matlab\RelOrbSim\PredictionAccuracyTestCampaign";
CSV_name = "ModelDiscoveryAccuracyTestCampaign";

FileName_String = CSV_name + " - " + num2str(TestCampaign_ID) + ".csv";
Fin_Dir = fullfile(Data_CSV_dir, FileName_String);

%% Import CSV and Parse Into Array
Raw_Data = csvread(Fin_Dir);

Ecc = Raw_Data(:,1);
Theta0 = Raw_Data(:,2);
Ts = Raw_Data(:,3);
Tlin = Raw_Data(:,4);
MSE_train = Raw_Data(:,5);
MSE_test = Raw_Data(:,6);
Py_time = Raw_Data(:,7);
Mat_time = Raw_Data(:,8);

%% Create Subsets based on Control Variable Selection

% Since there are 5 variables to plot in a 3d plot,
% 3 variables are 3 axis, 1 variable is color, 1 variable is selected for
% subseting other data

% Varibale for Selection: Eccentricity-subplots
% Variable for Color: Theta0-colorbar
% Variable for 3 axis: Tlin-X, Ts-Y, MSE_Test-Z 

% Plot
fig = figure('units','normalized','outerposition',[0 0 1 1]);

Ecc_unique = unique(Ecc);
font_size = 14;
marker_size = 50;
num_rows_subplot = 2;

for ii = 1:numel(Ecc_unique)    
    sp = subplot(num_rows_subplot, ceil(numel(Ecc_unique)/num_rows_subplot), ii );

    subset_idx = find(Ecc==Ecc_unique(ii));
    switch z_var
        case 'MSE_test'
            scatter3(Tlin(subset_idx), Ts(subset_idx), log10(MSE_test(subset_idx)), marker_size, Theta0(subset_idx), "filled")
            zlabel("$Log_{10}(pred.~MSE)$ [$m^2$]",'fontsize',font_size,'interpreter','latex')
            title_fig = "Model Prediction Test Campaign - " + num2str(TestCampaign_ID) + " MSE PRED";
        case 'MSE_train'
            scatter3(Tlin(subset_idx), Ts(subset_idx), log10(MSE_train(subset_idx)), marker_size, Theta0(subset_idx), "filled")
            zlabel("$Log_{10}(disc.~MSE)$ [$m^2$]",'fontsize',font_size,'interpreter','latex')
            title_fig = "Model Prediction Test Campaign - " + num2str(TestCampaign_ID) + " MSE DISC";
        case 'Py_time'
            scatter3(Tlin(subset_idx), Ts(subset_idx), log10(Py_time(subset_idx)), marker_size, Theta0(subset_idx), "filled")
            zlabel("$Log_{10}(Comp.~Time)$ [$s$]",'fontsize',font_size,'interpreter','latex')
            title_fig = "Model Prediction Test Campaign - " + num2str(TestCampaign_ID) + " COMP TIME";
    end
    subtitle("e = " + num2str(Ecc_unique(ii)),'fontsize',font_size,'interpreter','latex')
    xlabel("$\Delta$ $T_{lin}$ [$s$]",'fontsize',font_size,'interpreter','latex')
    ylabel("$T_{samp}$ [$s$]",'fontsize',font_size,'interpreter','latex')
    grid minor

    colormap(sp,"jet");
    
    switch view_type
        case '3d'
            view(3);
        case 'XY'
            view(0,90); % X-Y in view
        case 'YZ'
            view(90,0); % Y-Z in view
        case 'ZX'
            view(0,0); % Z-X in view
    end

end
sgtitle("Results: " + title_fig,'interpreter','latex')

h = axes(fig,'visible','off'); 
cb = colorbar(h,'Position',[0.93 0.168 0.022 0.7]); % attach colorbar to h
colormap(cb, 'jet');
caxis(h,[min(Theta0(:)),max(Theta0(:))]);             % set colorbar limits
cb.Label.String = "$\theta_0$ [$Rad$]";
cb.Label.Interpreter = 'latex';
cb.TickLabelInterpreter = 'latex';
cb.FontSize = font_size;



