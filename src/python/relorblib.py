# Custom Module For Relative Orbit Regression Functions
import pandas
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import rainbow
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches

import pysindy as ps

###############################################################################
def ImportSimCSV(ParentRelPath):
    FileNames = ('Times', 'Inputs', 'rho', 'rho_dot', 'rho_dotdot')
    with open((ParentRelPath + FileNames[0] + '.csv'), newline='') as csvfile:  
        times = np.array(list(csv.reader(csvfile))).astype(float)
    with open((ParentRelPath + FileNames[1] + '.csv'), newline='') as csvfile:  
        Inputs = np.array(list(csv.reader(csvfile))).astype(float)   
    with open((ParentRelPath + FileNames[2] + '.csv'), newline='') as csvfile:  
        rho = np.array(list(csv.reader(csvfile))).astype(float)
    with open((ParentRelPath + FileNames[3] + '.csv'), newline='') as csvfile:  
        rho_dot = np.array(list(csv.reader(csvfile))).astype(float)
    with open((ParentRelPath + FileNames[4] + '.csv'), newline='') as csvfile:  
        rho_dotdot = np.array(list(csv.reader(csvfile))).astype(float)
    return (times, Inputs, rho, rho_dot, rho_dotdot)
###############################################################################

###############################################################################
def dataSplit(X, X_dot, Inputs, T, N_splits):
    samples_per_split = np.floor(len(T)/N_splits)
    
    resized_total_samples = int((samples_per_split*N_splits))
    
    X_resized = X[0:resized_total_samples,:]
    X_dot_resized = X_dot[0:resized_total_samples,:]
    T_resized = T[0:resized_total_samples]
    Inputs_resized = Inputs[0:resized_total_samples,:]

    Split_X = np.split(X_resized, N_splits, axis=0)
    Split_X_dot = np.split(X_dot_resized, N_splits, axis=0)
    Split_Inputs = np.split(Inputs_resized, N_splits, axis=0)
    Split_T = np.split(T_resized, N_splits, axis=0)
    
    return (Split_X, Split_X_dot, Split_Inputs, Split_T)
###############################################################################

###############################################################################
def dataFit(x_train, t_train, Inputs_train, N_ensembles, x_dot_precomputed):
    dt = (t_train[1]-t_train[0])
    
    # threshold = 1e-12
    
    identity_library = ps.IdentityLibrary()
    identity_library.fit(np.concatenate((x_train, Inputs_train), axis=1))
    differentiation_method = ps.FiniteDifference(order=4)
    
    ###############################################################################
    n_features = identity_library.n_output_features_
    
    # Set constraints
    n_targets = x_train.shape[1]
    
    constraint_rhs = np.array([1,1,1,1,1,1,0,0,0])
    
    # One row per constraint, one column per coefficient
    constraint_lhs = np.zeros((constraint_rhs.size, n_targets * n_features))
    
    # For vx = ........
    constraint_lhs[0, 3+0*n_features] = 1.0 # vx
    # For vy = ........
    constraint_lhs[1, 4+1*n_features] = 1.0 # vy
    # For vz = ........
    constraint_lhs[2, 5+2*n_features] = 1.0 # vz
    # For ax = ........
    constraint_lhs[3, 6+3*n_features] = 1.0 # ux
    # For ay = ........
    constraint_lhs[4, 7+4*n_features] = 1.0 # uy
    # For az = ........
    constraint_lhs[5, 8+5*n_features] = 1.0 # uz
    # For Combined Constraints
    # 1*C(ax/y) + 1*C(ay/x) = 0
    constraint_lhs[6, 1+3*n_features] = 1.0 # y
    constraint_lhs[6, 0+4*n_features] = 1.0 # x
    # 1*C(ax/vy) + 1*C(ay/vx) = 0
    constraint_lhs[7, 4+3*n_features] = 1.0 # vy
    constraint_lhs[7, 3+4*n_features] = 1.0 # vx
    # 1*C(ax/x) + -1*C(ay/y) + 3*C(az/z)= 0
    constraint_lhs[8, 0+3*n_features] =  1.0 # x
    constraint_lhs[8, 1+4*n_features] = -1.0 # y
    constraint_lhs[8, 2+3*n_features] =  3.0 # z
    
    s = 1e30;
    # Each row corresponds to a measurement variable and each column to a function 
    # from the feature library
    # States                  x       y       z       dx      dy      dx     ux     uy     uz
    a_thresholds = np.array([[s,      s,      s,      0,      s,      s,     s,     s,      s],       # vx
                             [s,      s,      s,      s,      0,      s,     s,     s,      s],       # vy
                             [s,      s,      s,      s,      s,      0,     s,     s,      s],       # vz
                             [1e-11,  1e-11,  s,      s,      1e-11,  s,     0,     s,      s],       # ax
                             [1e-11,  1e-11,  s,      1e-11,  s,      s,     s,     0,      s],       # ay
                             [s,      s,      1e-11,  s,      s,      s,     s,     s,      0]     ]) # az
                             
    csr3_optimizer = ps.ConstrainedSR3(constraint_rhs=constraint_rhs, 
                                       constraint_lhs=constraint_lhs,
                                       thresholder="weighted_l1",
                                       tol=1e-9,
                                       # threshold=threshold,
                                       max_iter=10000,
                                       normalize_columns=False,
                                       # initial_guess=initial_guess,
                                       # trimming_fraction=0.5,
                                       fit_intercept=False,
                                       inequality_constraints = False,
                                       thresholds=a_thresholds,
    )
    
    # Fit The Model
    
    feature_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ux', 'uy', 'uz']
    
    model = ps.SINDy(feature_names = feature_names,
                     optimizer = csr3_optimizer,
                     feature_library = identity_library,
                     differentiation_method=differentiation_method,
    )
    
    model.fit(x_train, 
              t=dt, 
              # x_dot=x_dot_precomputed,
              u=Inputs_train,
              # library_ensemble=True,
              ensemble=True,
              n_models=N_ensembles,
              # n_candidates_to_drop=1, 
              # unbias=True
              # quiet=True
    )
    
    # Actual: 
        # ax -> 4.2e-6*x,   -4e-9*y,    2.4e-3*vy
        # ay -> 4e-9*x,     ~1e-7*y,    -2.4e-3*vx, 
        # az -> -1.45e-6*z

    # Approx Values:
        # kw^3/2 -> (<=) 1.4514... × 10^-6
        # w -> (<=) 1.20475531... × 10^-3
        # dw -> (<=) ?????? -4 x 10^-9

    # Formula
        # ax -> (2*kw3_2 + w^2)*x,          dw*y,                                       2*w*vy
        # ay ->      -dw*x,            (w^2 - kw3_2)*y,                   -2*w*vx, 
        # az ->                                              -(kw3_2)*z
    
    return model
###############################################################################

###############################################################################
def splitFit(X, X_dot, Inputs, T, N_splits, N_ensembles):
    (Split_X, Split_X_dot, Split_Inputs, Split_T) = dataSplit(X, X_dot, Inputs, T, N_splits)
    Models = []
    for split_idx in range(N_splits):
        Models.append(dataFit(Split_X[split_idx], Split_T[split_idx], Split_Inputs[split_idx], N_ensembles, Split_X_dot[split_idx]))
    SplitData = [Split_X, Split_X_dot, Split_Inputs, Split_T]
    return (Models, SplitData)
###############################################################################

###############################################################################
def printSysA(Model_Coefs):
    #.set_printoptions(linewidth=200)
    column_labels = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ux', 'uy', 'uz']
    row_labels = ['dx/dt', 'dy/dt', 'dz/dt', 'dvx/dt', 'dvy/dt', 'dvz/dt']
    pandas.set_option("display.precision", 1)
    Sys_A = pandas.DataFrame(Model_Coefs, columns=column_labels, index=row_labels)
    print("")
    print("Estimated System LTI Matrix --> Ensemble Averaged:")
    print(Sys_A)
    print("")
###############################################################################

###############################################################################
def printEnsembleParams(est_Params):
    n_models = len(est_Params[0])
    column_labels = [str(x+1) for x in range(n_models)]
    row_labels = ['kw3_2', 'dw', 'w']
    pandas.set_option("display.precision", 1)
    est_P = pandas.DataFrame(est_Params, columns=column_labels, index=row_labels)
    print("")
    print("Estimated Parameters --> From each LTI Sys model in ensembling:")
    print(est_P)
    print("")
###############################################################################

###############################################################################
def estimate_params(ensemble_coefs):
    n_models = len(ensemble_coefs)
    
    kw3_2_from_az = [0]*n_models
    dw_avg_from_axay = [0]*n_models
    w_from_axay = [0]*n_models
    w_from_ax2 = [0]*n_models
    w_from_ay2 = [0]*n_models
    w_avg = [0]*n_models
    
    for x in range(n_models):
        kw3_2_from_az[x] = -ensemble_coefs[1][5,2]
        dw_avg_from_axay[x] = ( ensemble_coefs[1][3,1] - ensemble_coefs[1][4,0] )/2
        w_from_axay[x] = ( ensemble_coefs[1][3,4] - ensemble_coefs[1][4,3] )/4
        w_from_ax2[x] = np.sqrt( ensemble_coefs[1][3,0] + 2*(ensemble_coefs[1][5,2]) )
        w_from_ay2[x] = np.sqrt( ensemble_coefs[1][3,0] - ensemble_coefs[1][5,2] )
        w_avg[x] = np.sum([w_from_axay[x], w_from_ax2[x], w_from_ay2[x]])/3

    est_Params = [kw3_2_from_az, dw_avg_from_axay, w_avg]
    
    # Print the results
    printEnsembleParams(est_Params)

    return est_Params
###############################################################################

###############################################################################
# def estimate_split_params(Split_Models):
#     n_models = len(Split_Models)
    
#     kw3_2_from_az = [0]*n_models
#     dw_avg_from_axay = [0]*n_models
#     w_from_axay = [0]*n_models
#     w_from_ax2 = [0]*n_models
#     w_from_ay2 = [0]*n_models
#     w_avg = [0]*n_models
    
#     for x in range(n_models):
#         mdl_coefs = Split_Models[x].coefficients()
        
#         kw3_2_from_az[x] = -mdl_coefs[5,2]
#         dw_avg_from_axay[x] = ( mdl_coefs[3,1] - mdl_coefs[4,0] )/2
#         w_from_axay[x] = ( mdl_coefs[3,4] - mdl_coefs[4,3] )/4
#         w_from_ax2[x] = np.sqrt( np.abs( mdl_coefs[3,0] + 2*(mdl_coefs[5,2]) ) )
#         w_from_ay2[x] = np.sqrt( np.abs( mdl_coefs[3,0] - mdl_coefs[5,2] ) )
        
#         w_avg[x] = np.sum([w_from_axay[x], w_from_ax2[x], w_from_ay2[x]])/3

#     est_Params = [kw3_2_from_az, dw_avg_from_axay, w_avg]
    
#     # Print the results
#     printEnsembleParams(est_Params)

#     return est_Params
def estimate_split_params(Split_Models):
    n_models = len(Split_Models)
    
    kw3_2_from_az = [0]*n_models
    dw_avg_from_axay = [0]*n_models
    w_from_axay = [0]*n_models
    w_from_ax2 = [0]*n_models
    w_from_ay2 = [0]*n_models
    
    for x in range(n_models):
        mdl_coefs = Split_Models[x].coefficients()
        
        kw3_2_from_az[x] = -mdl_coefs[5,2]
        dw_avg_from_axay[x] = ( mdl_coefs[3,1] - mdl_coefs[4,0] )/2
        w_from_axay[x] = ( mdl_coefs[3,4] - mdl_coefs[4,3] )/4
        w_from_ax2[x] = np.sqrt( np.abs( mdl_coefs[3,0] + 2*(mdl_coefs[5,2]) ) )
        w_from_ay2[x] = np.sqrt( np.abs( mdl_coefs[3,0] - mdl_coefs[5,2] ) )

    est_Params = [kw3_2_from_az, dw_avg_from_axay, w_from_axay, w_from_ax2, w_from_ay2]

    return est_Params
###############################################################################

###############################################################################
def plot_est_params(est_Params):
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines.right.set_position(("axes", 1.15))
    
    # fig.suptitle("Estimated Parameters", fontsize=14)
    fig.subplots_adjust(left=0.5)
        
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    n_models = len(est_Params[1])
    X_vals = np.linspace(1,n_models,n_models)
    lns1 = ax1.plot(X_vals[0:len(est_Params[0])], est_Params[0], "r", label="$kw^{1.5}$", linewidth=2)
    lns2 = ax2.plot(X_vals[0:len(est_Params[1])], est_Params[1], "b", label="$dw$", alpha=0.4, linewidth=2)
    lns3 = ax3.plot(X_vals[0:len(est_Params[2])], est_Params[2], "g", label="$w$", alpha=0.4, linewidth=2)
    
    ax1.set_title('Estimated Prameters From Linearised Models')
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    
    ax1.set(xlabel="Linearisation Index [-]", ylabel="Estimated Values: $kw^{1.5}$")
    ax2.set(ylabel="Estimated Values: $dw$")
    ax3.set(ylabel="Estimated Values: $w$")
###############################################################################

###############################################################################
def model_simulate(x, t, u, model, model_num):
    dt = (t[1]-t[0])
    x0 = x[0, :]
    
    # print("Model " + str(model_num+1) + ' score = ' + str(model.score(x, t=dt, u=u)) )
    
    x_sim = model.simulate(x0, t=t, u=u)
    t_sim = t[:-1]
    return(x_sim, t_sim)
###############################################################################

###############################################################################
def plot_split_model_simulation(X, X_dot, Inputs, T, Split_Data, Split_Models, N_ensembles):
    N_models = len(Split_Models)
    dt = T[1]-T[0]
    
    Split_X = Split_Data[0]
    Split_X_dot = Split_Data[1]
    Split_Inputs = Split_Data[2]
    Split_T = Split_Data[3]
    
    x_sim = []
    t_sim = []
    x = []
    t = []
    u = []
    
    for model_idx in range(N_models):
        (x_sim_temp, t_sim_temp) = model_simulate(Split_X[model_idx], Split_T[model_idx], Split_Inputs[model_idx], Split_Models[model_idx], model_idx)
        x_sim.append(x_sim_temp)
        t_sim.append(t_sim_temp)
        x.append(Split_X[model_idx][0:-1, :])
        t.append(Split_T[model_idx][0:-1])
        u.append(Split_Inputs[model_idx][0:-1, :])
        
    x_sim = np.vstack(x_sim)
    t_sim = np.hstack(t_sim)

    x = np.vstack(x)
    t = np.hstack(t)
    u = np.vstack(u)
    
    plot_kws = dict(linewidth=2)
    plot_title = 'Model Identification Results'
    common_title = plot_title +' --> SampleTime: '+str(dt)+'; N_segments: '+str(N_models)+'; N_Ensemble: '+str(N_ensembles)

    red_patch = mpatches.Patch(color='red', label='x - Radial')
    blue_patch = mpatches.Patch(color='blue', label='y - Along Track')
    green_patch = mpatches.Patch(color='green', label='z - Cross Track')
    
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(common_title, fontsize=14)
    fig.legend(['x', 'y', 'z'], handles=[red_patch, blue_patch, green_patch], loc='lower center',\
               frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.10))
    
    axs[0].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    axs[0].plot(t, x[:, 0], "r", label="$Data$", **plot_kws)
    axs[0].plot(t, x[:, 1], "b", alpha=0.4, **plot_kws)
    axs[0].plot(t, x[:, 2], "g", alpha=0.4, **plot_kws)
    axs[0].plot(t_sim, x_sim[:, 0], "k--", label="SINDy", **plot_kws)
    axs[0].plot(t_sim, x_sim[:, 1], "k--")
    axs[0].plot(t_sim, x_sim[:, 2], "k--")
    axs[0].set_title('Relative Position')
    axs[0].legend()
    axs[0].set(xlabel="t [s]", ylabel="$r_k [km]$")
    axs[0].legend(frameon=False, loc='upper center', ncol=3)
    # axes[0].grid(True)
    
    axs[1].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    axs[1].plot(t, x[:, 3], "r", label="$Data$", **plot_kws)
    axs[1].plot(t, x[:, 4], "b", alpha=0.4, **plot_kws)
    axs[1].plot(t, x[:, 5], "g", alpha=0.4, **plot_kws)
    axs[1].plot(t_sim, x_sim[:, 3], "k--", label="SINDy", **plot_kws)
    axs[1].plot(t_sim, x_sim[:, 4], "k--")
    axs[1].plot(t_sim, x_sim[:, 5], "k--")
    axs[1].set_title('Relative Velocity')
    axs[1].legend()
    axs[1].set(xlabel="t [s]", ylabel="$V_k [m/s]$")
    axs[1].legend(frameon=False, loc='upper center', ncol=3)
    
    #########################
    fig = plt.figure(figsize=(26, 10))
    fig.suptitle(common_title, fontsize=14)
    
    ax = plt.axes(projection='3d')
    ax.plot3D(x[:, 1], x[:, 2], x[:, 3], "r", label="$Data$", **plot_kws)
    ax.plot3D(x_sim[:, 1], x_sim[:, 2], x_sim[:, 3], "k--", label="$Data$", **plot_kws)
    
    ax.legend()
    ax.set(xlabel="$x - radial [km]$", 
           ylabel="$y - along track [km]$", 
           zlabel="$z - normal [km]$")
    
    #########################
    
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(common_title, fontsize=14)
    fig.legend(['x', 'y', 'z'], handles=[red_patch, blue_patch, green_patch], loc='lower center',\
                frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.10))
        
    axs[0].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    axs[0].plot(t_sim, np.abs(x[:, 0]-x_sim[:, 0]), "r", label="$SINDy$", **plot_kws)
    axs[0].plot(t_sim, np.abs(x[:, 1]-x_sim[:, 1]), "b", alpha=0.4, **plot_kws)
    axs[0].plot(t_sim, np.abs(x[:, 2]-x_sim[:, 2]), "g", alpha=0.4, **plot_kws)
    axs[0].set_title('Relative Position - Absolute Error')
    axs[0].legend()
    axs[0].set(xlabel="t [s]", ylabel="$r_k [km]$")
    axs[0].legend(frameon=False, loc='lower center', ncol=2)
    axs[0].set_yscale('log')
    # axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    axs[1].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    axs[1].plot(t_sim, np.abs(x[:, 3]-x_sim[:, 3]), "r", label="$SINDy$", **plot_kws)
    axs[1].plot(t_sim, np.abs(x[:, 4]-x_sim[:, 4]), "b", alpha=0.4, **plot_kws)
    axs[1].plot(t_sim, np.abs(x[:, 5]-x_sim[:, 5]), "g", alpha=0.4, **plot_kws)
    axs[1].set_title('Relative Velocity - Absolute Error')
    axs[1].legend()
    axs[1].set(xlabel="t [s]", ylabel="$V_k [m/s]$")
    axs[1].legend(frameon=False, loc='lower center', ncol=2)
    axs[1].set_yscale('log')
    
    # control input plot
    fig = plt.figure(figsize=(13, 5))
    fig.suptitle(common_title, fontsize=14)
    ax = plt.axes()
    fig.legend(['ux', 'uy', 'uz'], handles=[red_patch, blue_patch, green_patch], loc='lower center',\
                frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.10))
        
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax.plot(t, u[:,0], "r", label="$U-Train$", **plot_kws)
    ax.plot(t, u[:,1], "b", alpha=0.4, **plot_kws)
    ax.plot(t, u[:,2], "g", alpha=0.4, **plot_kws)
    
    ax.set_title('Control Input')
    
    ax.legend()
    ax.set(xlabel="t [s]", ylabel="$u_k [km/s^2]$")
    ax.legend(frameon=False, loc='lower center', ncol=2)
###############################################################################

###############################################################################
def plot_split_model_simulation(X, X_dot, Inputs, T, Split_Data, Split_Models, N_ensembles):
    N_models = len(Split_Models)
    dt = T[1]-T[0]
    
    Split_X = Split_Data[0]
    Split_X_dot = Split_Data[1]
    Split_Inputs = Split_Data[2]
    Split_T = Split_Data[3]
    
    x_sim = []
    t_sim = []
    x = []
    t = []
    u = []
    
    for model_idx in range(N_models):
        (x_sim_temp, t_sim_temp) = model_simulate(Split_X[model_idx], Split_T[model_idx], Split_Inputs[model_idx], Split_Models[model_idx], model_idx)
        x_sim.append(x_sim_temp)
        t_sim.append(t_sim_temp)
        x.append(Split_X[model_idx][0:-1, :])
        t.append(Split_T[model_idx][0:-1])
        u.append(Split_Inputs[model_idx][0:-1, :])
        
    x_sim = np.vstack(x_sim)
    t_sim = np.hstack(t_sim)

    x = np.vstack(x)
    t = np.hstack(t)
    u = np.vstack(u)
    
    plot_kws = dict(linewidth=2)
    plot_title = 'Model Identification Results'
    common_title = plot_title +' --> SampleTime: '+str(dt)+'; N_segments: '+str(N_models)+'; N_Ensemble: '+str(N_ensembles)

    red_patch = mpatches.Patch(color='red', label='x - Radial')
    blue_patch = mpatches.Patch(color='blue', label='y - Along Track')
    green_patch = mpatches.Patch(color='green', label='z - Cross Track')
    
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(common_title, fontsize=14)
    fig.legend(['x', 'y', 'z'], handles=[red_patch, blue_patch, green_patch], loc='lower center',\
               frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.10))
    
    axs[0].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    axs[0].plot(t, x[:, 0], "r", label="$Data$", **plot_kws)
    axs[0].plot(t, x[:, 1], "b", alpha=0.4, **plot_kws)
    axs[0].plot(t, x[:, 2], "g", alpha=0.4, **plot_kws)
    axs[0].plot(t_sim, x_sim[:, 0], "k--", label="SINDy", **plot_kws)
    axs[0].plot(t_sim, x_sim[:, 1], "k--")
    axs[0].plot(t_sim, x_sim[:, 2], "k--")
    axs[0].set_title('Relative Position')
    axs[0].legend()
    axs[0].set(xlabel="t [s]", ylabel="$r_k [km]$")
    axs[0].legend(frameon=False, loc='upper center', ncol=3)
    # axes[0].grid(True)
    
    axs[1].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    axs[1].plot(t, x[:, 3], "r", label="$Data$", **plot_kws)
    axs[1].plot(t, x[:, 4], "b", alpha=0.4, **plot_kws)
    axs[1].plot(t, x[:, 5], "g", alpha=0.4, **plot_kws)
    axs[1].plot(t_sim, x_sim[:, 3], "k--", label="SINDy", **plot_kws)
    axs[1].plot(t_sim, x_sim[:, 4], "k--")
    axs[1].plot(t_sim, x_sim[:, 5], "k--")
    axs[1].set_title('Relative Velocity')
    axs[1].legend()
    axs[1].set(xlabel="t [s]", ylabel="$V_k [m/s]$")
    axs[1].legend(frameon=False, loc='upper center', ncol=3)
    
    #########################
    fig = plt.figure(figsize=(26, 10))
    fig.suptitle(common_title, fontsize=14)
    
    ax = plt.axes(projection='3d')
    ax.plot3D(x[:, 1], x[:, 2], x[:, 3], "r", label="$Data$", **plot_kws)
    ax.plot3D(x_sim[:, 1], x_sim[:, 2], x_sim[:, 3], "k--", label="$Data$", **plot_kws)
    
    ax.legend()
    ax.set(xlabel="$x - radial [km]$", 
           ylabel="$y - along track [km]$", 
           zlabel="$z - normal [km]$")
    
    #########################
    
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(common_title, fontsize=14)
    fig.legend(['x', 'y', 'z'], handles=[red_patch, blue_patch, green_patch], loc='lower center',\
                frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.10))
        
    axs[0].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    axs[0].plot(t_sim, np.abs(x[:, 0]-x_sim[:, 0]), "r", label="$SINDy$", **plot_kws)
    axs[0].plot(t_sim, np.abs(x[:, 1]-x_sim[:, 1]), "b", alpha=0.4, **plot_kws)
    axs[0].plot(t_sim, np.abs(x[:, 2]-x_sim[:, 2]), "g", alpha=0.4, **plot_kws)
    axs[0].set_title('Relative Position - Absolute Error')
    axs[0].legend()
    axs[0].set(xlabel="t [s]", ylabel="$r_k [km]$")
    axs[0].legend(frameon=False, loc='lower center', ncol=2)
    axs[0].set_yscale('log')
    # axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    axs[1].ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    axs[1].plot(t_sim, np.abs(x[:, 3]-x_sim[:, 3]), "r", label="$SINDy$", **plot_kws)
    axs[1].plot(t_sim, np.abs(x[:, 4]-x_sim[:, 4]), "b", alpha=0.4, **plot_kws)
    axs[1].plot(t_sim, np.abs(x[:, 5]-x_sim[:, 5]), "g", alpha=0.4, **plot_kws)
    axs[1].set_title('Relative Velocity - Absolute Error')
    axs[1].legend()
    axs[1].set(xlabel="t [s]", ylabel="$V_k [m/s]$")
    axs[1].legend(frameon=False, loc='lower center', ncol=2)
    axs[1].set_yscale('log')
    
    # control input plot
    fig = plt.figure(figsize=(13, 5))
    fig.suptitle(common_title, fontsize=14)
    ax = plt.axes()
    fig.legend(['ux', 'uy', 'uz'], handles=[red_patch, blue_patch, green_patch], loc='lower center',\
                frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.10))
        
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    ax.plot(t, u[:,0], "r", label="$U-Train$", **plot_kws)
    ax.plot(t, u[:,1], "b", alpha=0.4, **plot_kws)
    ax.plot(t, u[:,2], "g", alpha=0.4, **plot_kws)
    
    ax.set_title('Control Input')
    
    ax.legend()
    ax.set(xlabel="t [s]", ylabel="$u_k [km/s^2]$")
    ax.legend(frameon=False, loc='lower center', ncol=2)
###############################################################################

###############################################################################
def paramsFit(x_train, t_train, Inputs_train, N_ensembles, x_dot_precomputed):
    dt = (t_train[1]-t_train[0])
    
    # threshold = 1e-12
    
    identity_library = ps.IdentityLibrary()
    identity_library.fit(np.concatenate((x_train, Inputs_train), axis=1))
    differentiation_method = ps.FiniteDifference(order=4)
    
    ###############################################################################
    n_features = identity_library.n_output_features_
    
    # Set constraints
    n_targets = x_train.shape[1]
    
    constraint_rhs = np.array([1,1,1,1,1,1,0,0,0])
    
    # One row per constraint, one column per coefficient
    constraint_lhs = np.zeros((constraint_rhs.size, n_targets * n_features))
    
    # For vx = ........
    constraint_lhs[0, 3+0*n_features] = 1.0 # vx
    # For vy = ........
    constraint_lhs[1, 4+1*n_features] = 1.0 # vy
    # For vz = ........
    constraint_lhs[2, 5+2*n_features] = 1.0 # vz
    # For ax = ........
    constraint_lhs[3, 6+3*n_features] = 1.0 # ux
    # For ay = ........
    constraint_lhs[4, 7+4*n_features] = 1.0 # uy
    # For az = ........
    constraint_lhs[5, 8+5*n_features] = 1.0 # uz
    # For Combined Constraints
    # 1*C(ax/y) + 1*C(ay/x) = 0
    constraint_lhs[6, 1+3*n_features] = 1.0 # y
    constraint_lhs[6, 0+4*n_features] = 1.0 # x
    # 1*C(ax/vy) + 1*C(ay/vx) = 0
    constraint_lhs[7, 4+3*n_features] = 1.0 # vy
    constraint_lhs[7, 3+4*n_features] = 1.0 # vx
    # 1*C(ax/x) + -1*C(ay/y) + 3*C(az/z)= 0
    constraint_lhs[8, 0+3*n_features] =  1.0 # x
    constraint_lhs[8, 1+4*n_features] = -1.0 # y
    constraint_lhs[8, 2+3*n_features] =  3.0 # z
    
    s = 1e30;
    # Each row corresponds to a measurement variable and each column to a function 
    # from the feature library
    # States                  x       y       z       dx      dy      dx     ux     uy     uz
    a_thresholds = np.array([[s,      s,      s,      0,      s,      s,     s,     s,      s],       # vx
                             [s,      s,      s,      s,      0,      s,     s,     s,      s],       # vy
                             [s,      s,      s,      s,      s,      0,     s,     s,      s],       # vz
                             [1e-11,  1e-11,  s,      s,      1e-11,  s,     0,     s,      s],       # ax
                             [1e-11,  1e-11,  s,      1e-11,  s,      s,     s,     0,      s],       # ay
                             [s,      s,      1e-11,  s,      s,      s,     s,     s,      0]     ]) # az
                             
    csr3_optimizer = ps.ConstrainedSR3(constraint_rhs=constraint_rhs, 
                                       constraint_lhs=constraint_lhs,
                                       thresholder="weighted_l1",
                                       tol=1e-9,
                                       # threshold=threshold,
                                       max_iter=10000,
                                       normalize_columns=False,
                                       # initial_guess=initial_guess,
                                       # trimming_fraction=0.5,
                                       fit_intercept=False,
                                       inequality_constraints = False,
                                       thresholds=a_thresholds,
    )
    
    # Fit The Model
    
    feature_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ux', 'uy', 'uz']
    
    model = ps.SINDy(feature_names = feature_names,
                     optimizer = csr3_optimizer,
                     feature_library = identity_library,
                     differentiation_method=differentiation_method,
    )
    
    model.fit(x_train, 
              t=dt, 
              # x_dot=x_dot_precomputed,
              u=Inputs_train,
              # library_ensemble=True,
              ensemble=True,
              n_models=N_ensembles,
              # n_candidates_to_drop=1, 
              # unbias=True
              # quiet=True
    )
    
    # Actual: 
        # ax -> 4.2e-6*x,   -4e-9*y,    2.4e-3*vy
        # ay -> 4e-9*x,     ~1e-7*y,    -2.4e-3*vx, 
        # az -> -1.45e-6*z

    # Approx Values:
        # kw^3/2 -> (<=) 1.4514... × 10^-6
        # w -> (<=) 1.20475531... × 10^-3
        # dw -> (<=) ?????? -4 x 10^-9

    # Formula
        # ax -> (2*kw3_2 + w^2)*x,          dw*y,                                       2*w*vy
        # ay ->      -dw*x,            (w^2 - kw3_2)*y,                   -2*w*vx, 
        # az ->                                              -(kw3_2)*z
    
    return model
###############################################################################

###############################################################################
def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    import scipy
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    def sinfunc(t, A, w, p, c):  return A * np.sin(w*t + p) + c
    
    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w/(2.*np.pi)
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}
###############################################################################

###############################################################################

###############################################################################

###############################################################################
def combined_estimate(mu, sigma2):
    mu = np.array(mu); sigma2 = np.array(sigma2)
    combined_mu = np.sum(np.divide(mu, sigma2)) / np.sum(1/sigma2)
    combined_sigma2 = 1 / np.sum(1/sigma2)
    return [combined_mu, combined_sigma2]
###############################################################################

###############################################################################
def movmean(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    mov_mean = ret[n - 1:] / n
    return mov_mean
###############################################################################

###############################################################################
def movvar(a, n):
    from numpy.lib.stride_tricks import sliding_window_view
    a = np.array(a)
    windows = sliding_window_view(a, window_shape = n)
    mov_var = [0]*windows.shape[0]
    for i in range(windows.shape[0]):
        mov_var[i] = np.var(windows[i,:])
    return mov_var
###############################################################################

###############################################################################
def combined_mov_estimate(measures, n):
    n_measures = len(measures)

    mov_mean = [0]*n_measures;   mov_var = [0]*n_measures

    for i in range(n_measures):
        mov_mean[i] = movmean(measures[i], n)
        mov_var[i] = movvar(measures[i], n)
    
    mu = np.asarray(mov_mean)
    sigma2 = np.asarray(mov_mean)
    N_samples = mu.shape[1]
    
    combined_mu = []; combined_sigma2 = []
    
    for i in range(N_samples):
        [comb_mu, comb_sigma2] = combined_estimate(mu[:,i], sigma2[:,i])
        combined_mu = np.append(combined_mu, comb_mu)
        combined_sigma2 = np.append(combined_sigma2, comb_sigma2)
        
    return [combined_mu, combined_sigma2]
###############################################################################

###############################################################################
def plot_estimation_accuracy(measures, combined_mu):
    fig = plt.figure(figsize=(13, 5))
    ax = plt.axes()
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    ax.plot(measures[0], "r", label="$w --> ax2$", linewidth=2) # w_from_ax2
    ax.plot(measures[1], "b", label="$w --> ay2$", alpha=0.4, linewidth=2) # w_from_ay2
    ax.plot(combined_mu, "k", label="$w --> Estimated$", alpha=1, linewidth=2) # w from combination of Normal Distribution

    ax.set_title('Estimation of $w$ Through Combined Normal Distributions')
    ax.legend()
    ax.set(xlabel="Linearisation Index [-]", ylabel="w [rad/s]")
    ax.legend(frameon=False, loc='upper center', ncol=3)