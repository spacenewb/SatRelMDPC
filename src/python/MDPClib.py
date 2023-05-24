def model_discover():
    import csv
    import math
    import numpy as np
    import pysindy as ps
    from sklearn.metrics import mean_squared_error
    
    import warnings
    warnings.filterwarnings("ignore")
    
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
    
    MU = 3.98600433e+5 # From DE405
    
    # Convert to S.I Units if Flag is up
    if DataInSI_flag == 1:
        a_f_RTH = a_f_RTH*1e3
        rho = rho*1e3
        rho_dot = rho_dot*1e3
        rho_dotdot = rho_dotdot*1e3
        kep_T[:,0] = kep_T[:,0]*1e3
        kep_C[:,0] = kep_C[:,0]*1e3
        MU = 3.98600433e+14 # From DE405
    
    Times = times.reshape(-1)
    DT = Times[1] - Times[0]
    
    #%% First Guess - Estimated State Matrix Coefficients
    # Estimate Params from State Measurements
    rho_dotdot_homogenous = rho_dotdot - a_f_RTH
    
    # Experimental Assumption --> "h" is known
    est_kw32 = -1*np.divide(rho_dotdot_homogenous[:,2], rho[:,2])
    P_Chaser_est = np.multiply( kep_C[:,0], ( 1 - np.multiply(kep_C[:,1], kep_C[:,1]) ) )
    H_Chaser_est = np.sqrt(MU*P_Chaser_est)
    W_est = np.divide(est_kw32, (MU / H_Chaser_est**(3/2)))**(2/3)
    W_dot_est = (-2/MU)*np.multiply(H_Chaser_est, est_kw32)
    
    measured_mean_kw32 = np.median(est_kw32)
    measured_mean_W = np.median(W_est)
    measured_mean_dW = np.median(W_dot_est)
    
    # Estimate of the matrix coefficients
    k1 = 2*measured_mean_kw32 + measured_mean_W**2   # (2*kw3_2 + w^2)
    k2 = measured_mean_W**2 - measured_mean_kw32     # ((w^2 - kw3_2)
    k3 = -measured_mean_kw32                         # -(kw3_2)
    k4 = measured_mean_dW                            # (dw)
    k5 = 2*measured_mean_W                           # (2*w)
    
    initial_guess = np.array([[    0,     0,     0,     1,     0,     0,     0,     0,     0],
                                    [    0,     0,     0,     0,     1,     0,     0,     0,     0],
                                    [    0,     0,     0,     0,     0,     1,     0,     0,     0],
                                    [   k1,    k4,     0,     0,     0,    k5,     1,     0,     0],
                                    [  -k4,    k2,     0,     0,   -k5,     0,     0,     1,     0],
                                    [    0,     0,    k3,     0,     0,     0,     0,     0,     1]])
    
    #%% Regression - pySINDy library creation
    
    # Structure the data arrays - State Vector and its Derivative
    X = np.concatenate((rho, rho_dot), axis=1)
    
    N_ensembles = 50
    
    identity_library = ps.IdentityLibrary()
    identity_library.fit(np.concatenate((X, a_f_RTH), axis=1))
    
    # differentiation_method = ps.FiniteDifference(order=4) # Good
    differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 5}, order=6) # Better
    
    #%% Regression - pySINDy Coefficient Matrix Constraints
    n_features = identity_library.n_output_features_
    
    # Set constraints
    n_targets = X.shape[1]
    
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
    
    #%% Regression - pySINDy Coefficient Matrix Term Dependent Thresholds
    s = math.inf;
    f = 1e-2;
    
    # Each row corresponds to a measurement variable and each column to a function from the feature library
    
    # States                  x       y       z       dx      dy      dz     ux     uy     uz
    a_thresholds = np.abs(np.array([[s,      s,      s,      0,      s,      s,     s,     s,      s],        # vx
                                    [s,      s,      s,      s,      0,      s,     s,     s,      s],        # vy
                                    [s,      s,      s,      s,      s,      0,     s,     s,      s],        # vz
                                    [k1*f,   k4*f,   s,      s,      k5*f,   s,     0,     s,      s],        # ax
                                    [k4*f,   k2*f,   s,      k5*f,   s,      s,     s,     0,      s],        # ay
                                    [s,      s,      k3*f,   s,      s,      s,     s,     s,      0]     ])) # az
    
    #%% Regression - pySINDy CSR3 Optimiser & Model Fit
    csr3_optimizer = ps.ConstrainedSR3(constraint_rhs=constraint_rhs, 
                                       constraint_lhs=constraint_lhs,
                                       thresholder="weighted_l1",
                                       nu=0.0000000001,
                                       tol=1e-16,
                                       max_iter=100,
                                       normalize_columns=False,
                                       initial_guess=initial_guess,
                                       fit_intercept=True,
                                       inequality_constraints = False,
                                       thresholds=a_thresholds,
    )
    # Fit The Model
    model = ps.SINDy(optimizer = csr3_optimizer,
                     feature_library = identity_library,
                     differentiation_method=differentiation_method,
    )
    model.fit(X, 
              t=DT, 
              u=a_f_RTH,
              ensemble=True,
              n_models=N_ensembles,
              unbias=True,
              quiet=True
    )
    
    Model_Coef_List = model.coef_list
    
    #%% System Matrix Ensemble Bagging - Weighted Average of Ensemble Matrix Coefficients (MSE_Discovery)
    # Bagging from Ensembles Coefficient List
    # Compute the average of the coefficients, weighted by the MSE & MAPE on the test data.
    MSE_Discovery = np.zeros(np.shape(Model_Coef_List)[0])
    for i in range(np.shape(Model_Coef_List)[0]):
        csr3_optimizer.coef_ = np.asarray(Model_Coef_List)[i, :, :]
        MSE_Discovery[i] = model.score(X, t=DT, u=a_f_RTH, metric=mean_squared_error)
    AugSysMat_W_MSE = np.average(Model_Coef_List, axis=0, weights=MSE_Discovery)
    SysMat = AugSysMat_W_MSE[:,:6]
    
    #%% Setup Output
    print(SysMat)
    return SysMat
    np.savetxt("C:/Users/vvh19/OneDrive/Documents/GitHub/SatRelMDPC/src/matlab/RelOrbSim/SysMat.csv", SysMat, delimiter=",")

model_discover()