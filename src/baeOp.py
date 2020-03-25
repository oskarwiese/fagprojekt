ACTIV = {0: torch.nn.Tanh(),
         1: torch.nn.ReLU(),
         2: torch.nn.ReLU6(),
         3: torch.nn.Sigmoid(),
        }

netsize = 256
## we have to define the function we want to maximize --> validation accuracy, 
## note it should take a 2D ndarray but it is ok that it assumes only one point
## in this setting
def objective_function(x): 
    MODEL = Model(categorical_embedding_sizes, 4, 1, [16,32,64,128,64], p=0.5)
    print(x)
    # we have to handle the categorical variables that is convert 0/1 to labels
    # log2/sqrt and gini/entropy
    
    param = x[0]
    hyperparameters = {
        'hidden_units_1': int(np.ceil(param[0]*netsize)),
        'hidden_units_2': int(np.ceil(param[1]*netsize)),
        'hidden_units_3': int(np.ceil(param[2]*netsize)),
        'hidden_units_4': int(np.ceil(param[3]*netsize)),
        'hidden_units_5': int(np.ceil(param[4]*netsize)),
        'p': param[5],
        'activation_func': ACTIV[int(param[6])]
    }
    print(hyperparameters)
    model, neural_acc  = train_model(hyperparameters, MODEL)
    print(neural_acc)
    return -neural_acc
    
# define the dictionary for GPyOpt
domain = [{'hidden_units_1'   : 'var_1', 'type': 'continuous', 'domain': (0 , 1)},
          {'hidden_units_2'   : 'var_1', 'type': 'continuous', 'domain': (0, 1)},
          {'hidden_units_3'   : 'var_1', 'type': 'continuous', 'domain': (0 , 1)},
          {'hidden_units_4'   : 'var_1', 'type': 'continuous', 'domain': (0 , 1)},
          {'hidden_units_5'   : 'var_1', 'type': 'continuous', 'domain': (0 , 1)},
          {'p'              : 'var_2', 'type': 'continuous',  'domain': (0 , 1)},
          {'activation_func': 'var_3', 'type': 'categorical','domain': tuple(np.arange(4))}]


opt = GPyOpt.methods.BayesianOptimization(f = objective_function,   # function to optimize
                                              domain = domain,         # box-constrains of the problem
                                              acquisition_type = "EI",      # Select acquisition function MPI, EI, LCB
                                             )
opt.acquisition.exploration_weight=.1

opt.run_optimization(max_iter = 100) 


x_best = opt.X[np.argmin(opt.Y)]
print("bedste: ", x_best)
#print("The best parameters obtained: n_estimators=" + str(x_best[0]) + ", max_depth=" + str(x_best[1]) + ", max_features=" + str(
#    x_best[2])  + ", criterion=" + str(
#    x_best[3]))