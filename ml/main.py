from ml.smac_examples.branin.branin import branin
from smac.facade.func_facade import fmin_smac


x, cost, _ = fmin_smac(func=branin,  # function
                       x0=[0, 0],  # default configuration
                       bounds=[(-5, 10), (0, 15)],  # limits
                       maxfun=10,  # maximum number of evaluations
                       rng=3)  # random seed

print("Optimum at {} with cost of {}".format(x, cost))