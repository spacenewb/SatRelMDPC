def EstimateSindyWithControl_MSE():
    import csv
    import math
    import time
    import numpy as np
    import pysindy as ps
    from sklearn.metrics import mean_squared_error
    
    import warnings
    warnings.filterwarnings("ignore")
    
    # get the start time
    st = time.process_time()
    
    CSV_dir = "C:/Users/vvh19/OneDrive/Documents/GitHub/SatRelMDPC/src/matlab/RelOrbSim/Export/"
    DataInSI_flag = 1

    # Import from CSV
    with open(CSV_dir + 'times.csv', newline='') as csvfile:  
        times = np.array(list(csv.reader(csvfile))).astype(float)
    with open(CSV_dir + 'a_f_RTH.csv', newline='') as csvfile:  
        a_f_RTH = np.array(list(csv.reader(csvfile))).astype(float)
    with open(CSV_dir + 'rho.csv', newline='') as csvfile:  
        rho = np.array(list(csv.reader(csvfile))).astype(float)
    with open(CSV_dir + 'rho_dot.csv', newline='') as csvfile:  
        rho_dot = np.array(list(csv.reader(csvfile))).astype(float)
    with open(CSV_dir + 'rho_dotdot.csv', newline='') as csvfile:  
        rho_dotdot = np.array(list(csv.reader(csvfile))).astype(float)
    with open(CSV_dir + 'kep_T.csv', newline='') as csvfile:  
        kep_T = np.array(list(csv.reader(csvfile))).astype(float)
    with open(CSV_dir + 'kep_C.csv', newline='') as csvfile:  
        kep_C = np.array(list(csv.reader(csvfile))).astype(float)
    with open(CSV_dir + 'params_T.csv', newline='') as csvfile:  
        params_T = np.array(list(csv.reader(csvfile))).astype(float)
    with open(CSV_dir + 'params_C.csv', newline='') as csvfile:  
        params_C = np.array(list(csv.reader(csvfile))).astype(float)
    
    feature_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'ux', 'uy', 'uz']
    target_names = ['dx/dt', 'dy/dt', 'dz/dt', 'dvx/dt', 'dvy/dt', 'dvz/dt']
    
    MU = 3.98600433e+5 # From DE405

    # Convert to S.I Units if Flag is up
    if DataInSI_flag == 1:
        a_f_RTH = a_f_RTH*1e3
        rho = rho*1e3
        rho_dot = rho_dot*1e3
        rho_dotdot = rho_dotdot*1e3
        kep_T[:,0] = kep_T[:,0]*1e3
        kep_C[:,0] = kep_C[:,0]*1e3
        MU = 3.98600433e+5 * 1e9 # From DE405
    
    # Divide Data into estimation and verification sets
    raw_data_samples = len(times);
    
    if np.remainder(raw_data_samples, 2) != 0:
        raw_times = times.reshape(-1)[:-1]
        raw_a_f_RTH = a_f_RTH[:-1,:]
        raw_rho = rho[:-1,:]
        raw_rho_dot = rho_dot[:-1,:]
        raw_rho_dotdot = rho_dotdot[:-1,:]
        raw_kep_T = kep_T[:-1,:]
        raw_kep_C = kep_C[:-1,:]
        raw_params_T = params_T[:-1,:]
        raw_params_C = params_C[:-1,:]
    else:
        raw_times = times.reshape(-1)
        raw_a_f_RTH = a_f_RTH
        raw_rho = rho
        raw_rho_dot = rho_dot
        raw_rho_dotdot = rho_dotdot
        raw_kep_T = kep_T
        raw_kep_C = kep_C
        raw_params_T = params_T
        raw_params_C = params_C  
    
    (times, times_valid) = np.split(raw_times, 2)
    (a_f_RTH, a_f_RTH_valid) = np.vsplit(raw_a_f_RTH, 2)
    (rho, rho_valid) = np.vsplit(raw_rho, 2)
    (rho_dot, rho_dot_valid) = np.vsplit(raw_rho_dot, 2)
    (rho_dotdot, rho_dotdot_valid) = np.vsplit(raw_rho_dotdot, 2)
    (kep_T, kep_T_valid) = np.vsplit(raw_kep_T, 2)
    (kep_C, kep_C_valid) = np.vsplit(raw_kep_C, 2)
    (params_T, params_T_valid) = np.vsplit(raw_params_T, 2)
    (params_C, params_C_valid) = np.vsplit(raw_params_C, 2)
    
    # Estimate Params from State Measurements
    rho_dotdot_homogenous = rho_dotdot - a_f_RTH

    # Experimental Assumption --> "h" is known
    est_kw32 = -1*np.divide(rho_dotdot_homogenous[:,2], rho[:,2])
    P_Chaser_est = np.multiply( kep_C[:,0], ( 1 - np.multiply(kep_C[:,1], kep_C[:,1]) ) )
    H_Chaser_est = np.sqrt(MU*P_Chaser_est)
    W_est = np.divide(est_kw32, (MU / H_Chaser_est**(3/2)))**(2/3)
    R_est = np.sqrt(np.divide(H_Chaser_est, W_est))

    Times = times.reshape(-1)
    DT = Times[1] - Times[0]
    R_dot_est = np.gradient(R_est, Times)
    W_dot_est = (-2/MU)*np.multiply(H_Chaser_est, est_kw32)

    est_kw32_mean = np.mean(est_kw32);     est_kw32_median = np.median(est_kw32)
    est_W_mean = np.mean(W_est);           est_W_median = np.median(W_est)
    est_dW_mean = np.mean(W_dot_est);      est_dW_median = np.median(W_dot_est)

    measured_mean_kw32 = est_kw32_median
    measured_mean_W = est_W_median
    measured_mean_dW = est_dW_median

    k1 = 2*measured_mean_kw32 + measured_mean_W**2   # (2*kw3_2 + w^2)
    k2 = measured_mean_W**2 - measured_mean_kw32     # ((w^2 - kw3_2)
    k3 = -measured_mean_kw32                         # -(kw3_2)
    k4 = measured_mean_dW                            # (dw)
    k5 = 2*measured_mean_W                           # (2*w)

    initial_guess_trial = np.array([[    0,     0,     0,     1,     0,     0,     0,     0,     0],
                                    [    0,     0,     0,     0,     1,     0,     0,     0,     0],
                                    [    0,     0,     0,     0,     0,     1,     0,     0,     0],
                                    [   k1,    k4,     0,     0,     0,    k5,     1,     0,     0],
                                    [  -k4,    k2,     0,     0,   -k5,     0,     0,     1,     0],
                                    [    0,     0,    k3,     0,     0,     0,     0,     0,     1]])

    # Estimation of System Matrix --> Estimate Params
    # Structure the data arrays
    X = np.concatenate((rho, rho_dot), axis=1)
    X_dot = np.concatenate((rho_dot, rho_dotdot), axis=1)
    T = times.reshape(-1) # 0 D Array

    # ToDo: Split data into train and test sets
    x_train = X
    t_train = T
    Inputs_train = a_f_RTH
    x_dot_precomputed = X_dot
    
    N_ensembles = 50

    dt = (t_train[1]-t_train[0])
    identity_library = ps.IdentityLibrary()
    identity_library.fit(np.concatenate((x_train, Inputs_train), axis=1))

    # differentiation_method = ps.FiniteDifference(order=4) # Good
    differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 5}, order=6) # Better

    n_features = identity_library.n_output_features_

    # Set constraints
    n_targets = x_train.shape[1]

    # constraint_rhs = np.array([1,1,1,1,1,1,0,0,0, -measured_mean_kw32, 2*measured_mean_W])
    constraint_rhs = np.array([1,1,1,1,1,1,0,0,0,k5])

    # One row per constraint, one column per coefficient
    constraint_lhs = np.zeros((constraint_rhs.size, n_targets * n_features))

    # Format:
    # constraint_lhs[constraint_number, coefficient_of_which_feature + contribution_to_which_target*n_features] = coefficient_factor
    # f:feature, t:target, C:coefficient --> C[f1/t3] = coefficient for contribution of f1 in t3

    # constraint_lhs[Constraint_number_in_[constraint_rhs], {f?} + {t?}*n_features] = coefficient_multiplier  # c1

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
    constraint_lhs[8, 2+5*n_features] =  3.0 # z

    # For 2W Constraint in ax
    constraint_lhs[9, 4+3*n_features] = 1.0 # vz

    s = math.inf;
    f = 1e-2;
    # Each row corresponds to a measurement variable and each column to a function 
    # from the feature library

    # States                  x       y       z       dx      dy      dz     ux     uy     uz
    a_thresholds = np.abs(np.array([[s,      s,      s,      0,      s,      s,     s,     s,      s],        # vx
                                    [s,      s,      s,      s,      0,      s,     s,     s,      s],        # vy
                                    [s,      s,      s,      s,      s,      0,     s,     s,      s],        # vz
                                    [k1*f,   k4*f,   s,      s,      k5*f,   s,     0,     s,      s],        # ax
                                    [k4*f,   k2*f,   s,      k5*f,   s,      s,     s,     0,      s],        # ay
                                    [s,      s,      k3*f,   s,      s,      s,     s,     s,      0]     ])) # az

    csr3_optimizer = ps.ConstrainedSR3(constraint_rhs=constraint_rhs, 
                                       constraint_lhs=constraint_lhs,
                                       thresholder="weighted_l1",
                                       nu=0.0000000001,
                                       tol=1e-16,
                                       max_iter=100,
                                       normalize_columns=False,
                                       initial_guess=initial_guess_trial,
                                       # trimming_fraction=0.1,
                                       fit_intercept=True,
                                       inequality_constraints = False,
                                       thresholds=a_thresholds,
    )

    # Fit The Model
    model = ps.SINDy(feature_names = feature_names,
                     optimizer = csr3_optimizer,
                     feature_library = identity_library,
                     differentiation_method=differentiation_method,
    )
    model.fit(x_train, 
              t=dt, 
              u=Inputs_train,
              ensemble=True,
              n_models=N_ensembles,
              # n_candidates_to_drop=1, 
              unbias=True
              # quiet=True
    )
    # Formula
        # ax -> (2*kw3_2 + w^2)*x,          dw*y,                0             0        2*w*vy,     1     0     0
        # ay ->      -dw*x,            (w^2 - kw3_2)*y,          0          -2*w*vx,       0        0     1     0
        # az ->        0                      0              -(kw3_2)*z,       0           0        0     0     1

    Model_Coef_List = model.coef_list
    Model_Coefs = model.coefficients()

    # SysA Bagging from Ensembles Coefficient List
    # Compute the average of the coefficients, weighted by the MSE & MAPE on the test data.
    MSE_Train = np.zeros(np.shape(Model_Coef_List)[0])
    for i in range(np.shape(Model_Coef_List)[0]):
        csr3_optimizer.coef_ = np.asarray(Model_Coef_List)[i, :, :]
        # For other metrics for scoring: https://scikit-learn.org/stable/modules/model_evaluation.html
        MSE_Train[i] = model.score(x_train, t=dt, u=Inputs_train, metric=mean_squared_error)
    SysA_W_Mse = np.average(Model_Coef_List, axis=0, weights=MSE_Train)
    
    # Validation
    X_valid = np.concatenate((rho_valid, rho_dot_valid), axis=1)
    X_dot_valid = np.concatenate((rho_dot_valid, rho_dotdot_valid), axis=1)
    T_valid = times_valid.reshape(-1) # 0 D Array
    x_test = X_valid
    t_test = T_valid
    Inputs_test = a_f_RTH_valid
    x_dot_precomputed_test = X_dot_valid
    
    csr3_optimizer.coef_ = np.asarray(SysA_W_Mse)
    MSE_Test = model.score(x_test, t=dt, u=Inputs_test, metric=mean_squared_error)
    
    # Define Outputs:
    Eccentricity = np.mean([kep_C[:,1], kep_T[:,1]])
    Theta0 = np.mean([kep_C[0,5], kep_T[0,5]])
    
    Eccentricity = np.mean([kep_C[:,1], kep_T[:,1]])
    Theta0 = kep_C[0,5]
    
    SampleTime = dt
    LinearisationInterval = Times[-1] - Times[0]
    BestMSE_Train = np.min(MSE_Train)
    
    # get the end time
    et = time.process_time()
    # calc CPU exec. time [s]
    res = et - st
    
    Result_names = ['Ecc', 'Theta0', 'Ts', 'LinInt', 'MSE_Train', 'MSE_Test', 'CPU Ex. Time']
    Results = [Eccentricity, Theta0, SampleTime, LinearisationInterval, BestMSE_Train, MSE_Test, res]
    
    for i in Results:
        print(i)
        
    return Results