# -*- coding: utf-8 -*-
#%%
###############################################################################
# NOTES:
# - IPOPT started searching outside the variable bounds, where some of our functions are no longer defined... so I hard-wired function evaluations for those regions for "disutil_inv", "disutil_inv_prime", and "disutil_inv_prime_2"
# - need to find a better way to illustrate solution, e.g. by plotting savings rates for up to six bbeta values over the range of ability or income levels
# - create simple function to compute agents' preferred savings rate (from EE)
###############################################################################
## housekeeping (open)
"""
IPOPT Tests

Description: Run preliminary tests on optimization problems using IPOPT.
Author: Chris Moser (Columbia University)
First created: March 18, 2018
Last edited: March 20, 2018
"""

## import modules
import numpy as np
from scipy.stats import beta, lognorm
from scipy.sparse import coo_matrix
from scipy.sparse import vstack as sparse_vstack
from math import exp, log, inf
import matplotlib.pyplot as plt
import pyipopt
import time
#from scipy.integrate import quad

## switches
resource_equality = 0 # 0 = allow surplus; 1 = force budget balance
print_intm = 0 # 0 = do nothing; 1 = print intermediate inputs
print_final = 1 # 0 = do nothing; 1 = print final inputs
check_tol = 10**(-5) # tolerance for feasibiltiy and IC check in final output
n_dec = 3 # number of decimals to print in final output
dist_family = 'uniform' # 'uniform' or 'beta_normal'
init = 'equal' # 'equal' or 'laissez_faire'

# set optimization parameters
almost_inf = 2.0*10.0**19 # inf or 2.0*10.0**19

## set model parameters
n_ttheta = 2
min_ttheta = .11
max_ttheta = .33
dist_ttheta_sigma = .75**.5
dist_ttheta_loc = 0
dist_ttheta_scale = 1
n_bbeta = 3
min_bbeta = 1.0
max_bbeta = 1.0
dist_bbeta_a = 2
dist_bbeta_b = 2
kkappa = 1.0
ddelta = 1.0
min_ddelta = 1.0
max_ddelta = 1.0
pareto_min = 2.0
pareto_max = 1.0
T_work = 1 # number of work periods
T_retire = 1 # number of retirement periods
R = 1.1 # net interest rate = 1 + r
eeta = 2 # reciprocal of consumption utility exponent (EIS)
ggamma = 1 # reciprocal of work disutility exponent (Frisch elasticity)
G = 0.0 # exogenous government spending level

## Python options
np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, 
                    nanstr='nan', precision=8, suppress=False, 
                    threshold=1000, formatter=None)

## user-defined functions
# disability level
def disabil(ttheta):
    """Disability level, inverse transformation of ability, D(ttheta)."""
    assert np.all(ttheta >= 0)
    d = 1/ttheta**(1 + 1/ggamma)
    return d

# consumption utility
def util(c):
    """Consumption utility function of CRRA form, u(c)."""
    assert np.all(c >= 0)
    if eeta != 1:
        u = (np.power(c, 1 - 1/eeta, dtype = float) - 1)/(1 - 1/eeta)
    else:
        u = log(c)
    return u

# first derivative of consumption utility
def util_prime(c):
    """First derivative of consumption utility function, u'(c)."""
    assert np.all(c >= 0)
    u = np.power(c, -1/eeta, dtype = float)
    return u

# second derivative of consumption utility
def util_prime_2(c):
    """Second derivative of consumption utility function, u''(c)."""
    assert np.all(c >= 0)
    u = -1/eeta*np.power(c, -1/eeta - 1, dtype = float)
    return u

# inverse consumption utility
def util_inv(u):
    """Inverse of consumption utility function, C(u)."""
    if eeta < 1:
        assert np.all(u <= -1/(1 - 1/eeta))
    elif eeta > 1:
        assert np.all(u >= -1/(1 - 1/eeta))
    if eeta != 1:
        c = np.power((1 - 1/eeta)*u + 1, 1/(1 - 1/eeta), dtype = float)
    else:
        c = exp(u)
    return c

# first derivative of inverse consumption utility
def util_inv_prime(u):
    """First derivative of inverse consumption utility function, C'(u)."""
    if eeta < 1:
        assert np.all(u <= -1/(1 - 1/eeta))
    elif eeta > 1:
        assert np.all(u >= -1/(1 - 1/eeta))
    if eeta != 1:
        c = np.power((1 - 1/eeta)*u + 1, (1/eeta)/(1 - 1/eeta), dtype = float)
    else:
        c = exp(u)
    return c

# second derivative of inverse consumption utility
def util_inv_prime_2(u):
    """Second derivative of inverse consumption utility function, C''(u)."""
    if eeta < 1:
        assert np.all(u <= -1/(1 - 1/eeta))
    elif eeta > 1:
        assert np.all(u >= -1/(1 - 1/eeta))
    if eeta != 1:
        c = (1/eeta)*np.power((1 - 1/eeta)*u + 1, (2/eeta - 1)/(1 - 1/eeta), dtype = float)
    else:
        c = exp(u)
    return c

# ouput disutility
def disutil(y):
    """Output disutility function, v(y)."""
    assert np.all(y >= 0)
    v = kkappa*np.power(y, 1 + 1/ggamma, dtype = float)/(1 + 1/ggamma)
    return v

# first derivative of ouput disutility
def disutil_prime(y):
    """First derivative of the output disutility function, v'(y)."""
    assert np.all(y >= 0)
    v = kkappa*np.power(y, 1/ggamma, dtype = float)
    return v

# second derivative of ouput disutility
def disutil_prime_2(y):
    """Second derivative of the output disutility function, v''(y)."""
    assert np.all(y >= 0)
    v = (kkappa/ggamma)*np.power(y, 1/ggamma - 1, dtype = float)
    return v

# inverse ouput disutility
def disutil_inv(v):
    """Inverse of the output disutility function, y = Y(v)."""
    assert np.all(v >= 0)
    y = np.power((1 + 1/ggamma)*v/kkappa, 1/(1 + 1/ggamma), dtype = float)
    return y

# first derivative of inverse ouput disutility
def disutil_inv_prime(v):
    """First derivative of the inverse of the output disutility function, y' = Y'(v)."""
    assert np.all(v >= 0)
    y = (1/kkappa)*np.power((1 + 1/ggamma)*v/kkappa, (-1/ggamma)/(1 + 1/ggamma), dtype = float)
    return y

# second derivative of inverse ouput disutility
def disutil_inv_prime_2(v):
    """First derivative of the inverse of the output disutility function, y'' = Y''(v)."""
    assert np.all(v >= 0)
    y = (-1/(kkappa**2*ggamma))*np.power((1 + 1/ggamma)*v/kkappa, (-2/ggamma - 1)/(1 + 1/ggamma), dtype = float)
    return y

## create type (beta, theta) vectors, type distributions, as well as a list and dictionary of types
ttheta = np.linspace(min_ttheta, max_ttheta, n_ttheta)
type_ttheta = np.repeat(ttheta, n_bbeta)
if print_intm: print('\n* THETA:\n ', type_ttheta)
d_ttheta = disabil(ttheta)
type_d_ttheta = np.repeat(d_ttheta, n_bbeta)
if print_intm: print('\n* D(THETA):\n ', type_ttheta)
bbeta = np.linspace(min_bbeta, max_bbeta, n_bbeta)
type_bbeta = np.concatenate([bbeta[:] for i in range(n_ttheta)])
if print_intm: print('\n* BETA:\n ', type_bbeta)
type_ddelta = np.repeat(ddelta, n_bbeta*n_ttheta)
#type_ddelta = np.repeat(ttheta, n_bbeta)
if print_intm: print('\n* DELTA:\n ', type_ddelta)
if dist_family == 'uniform':
    dist_types = np.array([1/(n_bbeta*n_ttheta)]*(n_bbeta*n_ttheta)) # XXX normalize so this integrates to 1?
elif dist_family == 'beta_normal':
    dist_bbeta = beta.pdf(bbeta, dist_bbeta_a, dist_bbeta_b)
    dist_ttheta = lognorm.pdf(ttheta, dist_ttheta_sigma, dist_ttheta_loc, dist_ttheta_scale)
    dist_types = np.kron(dist_bbeta, dist_ttheta) # XXX normalize so this integrates to 1?
type_set = [(i, j) for i in range(0, n_ttheta) for j in range(0, n_bbeta)]
if print_intm: print('\n* TYPE SET [(BETA, THETA)]:\n ', type_set)
n_type = len(type_set)
assert n_type == n_ttheta*n_bbeta
type_dict = dict(zip(range(len(type_set)), type_set))
if print_intm: print('\n* TYPE DICTIONARY [DICT: (BETA, THETA)]:\n ', type_dict)
n_var_per_type = T_work*2 + T_retire*1
n_var = n_var_per_type*n_type

## create choice variable bounds
if eeta < 1:
    u_lowerbar = -almost_inf
    u_upperbar = -1/(1 - 1/eeta)
elif eeta == 1:
    u_lowerbar = -almost_inf
    u_upperbar = +almost_inf
elif eeta > 1:
    u_lowerbar = -1/(1 - 1/eeta)
    u_upperbar = +almost_inf
v_lowerbar = 0
v_upperbar = +almost_inf
var_con_lhs = np.zeros(n_var)
var_con_rhs = np.zeros(n_var)
for p in range(n_type):
    var_con_lhs[3*p:3*p + 3] = [u_lowerbar, u_lowerbar, v_lowerbar]
    var_con_rhs[3*p:3*p + 3] = [u_upperbar, u_upperbar, v_upperbar]
if print_intm: print('\n* CHOICE VARIABLES LOWER BOUNDS:\n ', var_con_lhs)
if print_intm: print('\n* WORK DISUTILITY UPPER BOUNDS:\n ', var_con_rhs)

## create objective
pareto = np.linspace(pareto_min, pareto_max, n_ttheta)
type_pareto = np.kron(pareto, np.ones(n_bbeta))
if print_intm: print('\n* PARETO WEIGHTS:\n ', type_pareto)
obj_mat = np.concatenate([-dist_types[i]*type_pareto[i]*np.array([1, type_ddelta[i], -type_d_ttheta[i]]) for i in range(n_type)])
if print_intm: print('\n* OBJECTIVE:\n ', obj_mat)

## create list and dictionary of all IC constraints, and create IC constraints comatrix (one row of the matrix is one IC constraint, and an IC constraint is satisfied if <= 0) with bouns
ic_set = [(i, j) for i in range(n_type) for j in range(n_type) if i != j]
if print_intm: print('\n* IC CONSTRAINT SET [TYPE NUMBER --> TYPE NUMBER]:\n ', ic_set)
ic_set_ext = [(i, j) for i in type_set for j in type_set if i != j]
if print_intm: print('\n* EXTENSIVE IC CONSTRAINT SET [(BETA, THETA) --> (BETA, THETA)]:\n ', ic_set_ext)
n_ic = len(ic_set)
assert n_ic == n_type**2 - n_type
ic_dict = dict(zip(range(len(ic_set)), ic_set))
if print_intm: print('\n* IC CONSTRAINT DICTIONARY [DICT: (BETA, THETA) --> (BETA, THETA)]:\n ', ic_dict)
ic_rows = np.repeat([i for i in range(n_ic)], 2*n_var_per_type)
ic_cols = np.zeros(n_ic*2*n_var_per_type, dtype = 'int64')
ic_data = np.zeros(n_ic*2*n_var_per_type, dtype = 'float64')
loop_counter = 0
for i in range(n_type):
    for j in range(n_type):
        if i != j:
            ic_data[3*loop_counter:3*loop_counter + 6] = [-1, -type_bbeta[i]*type_ddelta[i], type_d_ttheta[i], 1, type_bbeta[i]*type_ddelta[i], -type_d_ttheta[i]]
            ic_cols[3*loop_counter:3*loop_counter + 3]= [3*i, 3*i + 1, 3*i + 2]
            loop_counter += 1
            ic_cols[3*loop_counter:3*loop_counter + 3] = [3*j, 3*j + 1, 3*j + 2]
            loop_counter += 1
ic_comat = coo_matrix((ic_data, (ic_rows, ic_cols)), shape=(n_ic, n_var))
if print_intm: print('\n* IC CONSTRAINTS COMATRIX CONTENTS:\n ', ic_comat)
#ic_comat_dict = dict(zip(range(n_ic), [ic_comat.toarray()[i, :] for i in range(n_ic)]))
#if print_intm: print('\n* IC CONSTRAINTS COMATRIX DICTIONARY:\n ', ic_comat_dict)
ic_con_lhs = np.repeat(-almost_inf, n_ic)
ic_con_rhs = np.zeros(n_ic, dtype = 'int64')
# XXX CREATE LOCAL IC MATRIX!

## nonlinear feasibility (resource) constraint
n_nonlin_con = 1
if resource_equality == 1:
    nonlin_con_lhs = np.array([0])
elif resource_equality == 0:
    nonlin_con_lhs = np.array([-almost_inf])
nonlin_con_rhs = np.array([0])
if print_intm: print('\n* FEASIBILITY CONSTRAINT LHS:\n ', nonlin_con_lhs)
if print_intm: print('\n* FEASIBILITY CONSTRAINT RHS:\n ', nonlin_con_rhs)

## stacked constraints
n_con = n_nonlin_con + n_ic
con_lhs = np.concatenate([nonlin_con_lhs, ic_con_lhs])
con_rhs = np.concatenate([nonlin_con_rhs, ic_con_rhs])

## define IPOPT problem functions
# objective function
def eval_f(x, user_data = None):
    """IPOPT: Objective function.
    
    INPUTS
    ------
    x: choice variable vector
    
    OUTPUT
    ------
    obj_mat@x: objective function, of dimension 1.
    """
    assert len(x) == len(obj_mat) == n_var
    return obj_mat@x

# gradient of the objective function
def eval_grad_f(x, user_data = None): # XXX check if IPOPT has special treatment for linear objectives
    """IPOPT: Gradient of the objective function.
    
    INPUTS
    ------
    x: choice variable vector
    
    OUTPUT
    ------
    obj_mat@x: gradient of the objective function, of dimension (1) x (number of variables).
    """
    assert len(x) == len(obj_mat) == n_var
    return obj_mat

# inequality constraints function
def eval_g(x, user_data = None):
    """IPOPT: Constraint function i.e. one nonlinear resource constraint and up to (N^2 - N) linear IC constraints.
    
    INPUTS
    ------
    x: choice variable vector
    
    OUTPUT
    ------
    jac_g: Constraint function, of dimension (1 + number of IC constraints) x (number of variables).
    """
    assert len(x) == ic_comat.shape[1] == n_var
    g = [(dist_types[i]*util_inv(x[3*i]), 
          dist_types[i]*util_inv(x[3*i + 1])/R, 
          -dist_types[i]*disutil_inv(x[3*i + 2])) for i in range(n_type)]
    return np.hstack([np.sum(g) + G, ic_comat@x])

# Jacobian (first derivative) matrix of the inequality constraints function
nnzj = n_var + 2*n_var_per_type*n_ic # IPOPT: number of nonzero entries in the Jacobian XXX change this later in active set loop!
def eval_jac_g(x, flag, user_data = None):
    """IPOPT: Jacobian of the constraint function.
    
    INPUTS
    ------
    x: choice variable vector
    flag: indiactor for requesting only row-column coordinates of nonzero entries
    
    OUTPUT
    ------
    jac_g: Jacobian of the constraint function, of dimension (number of constraints) x (number of variables).
    """
    if flag:
        return (np.concatenate([np.zeros(n_var, dtype = int), ic_rows + 1]), np.concatenate([range(n_var), ic_cols]))
    else:
        assert len(x) == ic_comat.shape[1] == n_var
        jac_g = np.concatenate([np.concatenate([(dist_types[i]*util_inv_prime(x[3*i]),
                                                 dist_types[i]*util_inv_prime(x[3*i + 1])/R,
                                                 -dist_types[i]*disutil_inv_prime(x[3*i + 2])) for i in range(n_type)]), ic_data])
        return jac_g

# Hessian (second derivative) matrix of the inequality constraints function
nnzh = n_var # IPOPT: number of nonzero entries in the Hessian (set nnzh=0 if Hessian not provided)
def eval_h(x, lagrange, obj_factor, flag, user_data = None):
    """IPOPT: Hessian of the Lagrangian associated with the objective and constraint function.
    
    INPUTS
    ------
    x: choice variable vector
    lagrange: vector of lagrange multipliers
    obj_factor: IPOPT-specific element in the Hessian, pre-multiplying the objective, but this drops out for linear objective functions like that considered here.
    flag: indicator for requesting only row-column coordinates of nonzero entries
    
    OUTPUT
    ------
    h: Hessian of the Lagrangian, of dimension (number of variables) x (number of variables).
    """
    if flag:
        return tuple([np.array(range(n_var), dtype=int) for i in range(2)])
    else:
        h = np.concatenate([(lagrange[0]*dist_types[i]*util_inv_prime_2(x[3*i]),
                             lagrange[0]*dist_types[i]*util_inv_prime_2(x[3*i + 1])/R,
                             -lagrange[0]*dist_types[i]*disutil_inv_prime_2(x[3*i + 2])) for i in range(n_type)])
        return h

# Indicator for whether Hessian is user-provided
def apply_new(x):
    """IPOPT: Callback function to indicate whether Hessian is user-provided.
    
    INPUTS
    ------
    x: choice variable vector
    
    OUTPUT
    ------
    True: always use user-provided Hessian.
    """
    return True

## create IPOPT problem
print('\n*** define IPOPT problem')
nlp = pyipopt.create(n_var, var_con_lhs, var_con_rhs, n_con, con_lhs, con_rhs, nnzj, nnzh, eval_f, eval_grad_f, eval_g, eval_jac_g, eval_h, apply_new)
nlp.int_option('max_iter',3000)
nlp.int_option('print_level',5)
#nlp.num_option('tol',1e-15)
#nlp.num_option('constr_viol_tol',1e-12)
#nlp.num_option('acceptable_tol',1e-15)
nlp.str_option('derivative_test','second-order')
##nlp.str_option('nlp_scaling_method','user-scaling')
##nlp.num_option('obj_scaling_factor',1.0)

## create initial guess
if init == 'equal': # XXX could do a lot better?!
    v_init = np.ones(n_type)
    y_init = disutil_inv(v_init)
    Y_init = sum(dist_types*y_init)
    C_npv_init = Y_init - G
    C_period_init = R*C_npv_init/(1+R)
    assert C_period_init + C_period_init/R == C_npv_init
    c_period_init = C_period_init/sum(dist_types)*np.ones(n_type)
    u_init = util(c_period_init)
    if resource_equality == 0:
        u_init = u_init - 10**(-4)
    x0 = np.concatenate([[u_init[i], u_init[i], v_init[i]] for i in range(n_type)])
elif init == 'laissez_faire':
    # XXX TBC!
    pass

## test runs
flag0_test = 0
flag1_test = 1
lagrange_test = np.ones(1)
obj_factor_test = 1
eval_f(x0)
eval_grad_f(x0)
eval_g(x0)
eval_jac_g(x0, flag0_test)
eval_jac_g(x0, flag1_test)
eval_h(x0, lagrange_test, obj_factor_test, flag0_test)
eval_h(x0, lagrange_test, obj_factor_test, flag1_test)
print('   * problem successfully initialized!')

## run IPOPT solver
print('\n*** run IPOPT optimization routine to solve problem')
time_start = time.time()
x, zl, zu, constraint_multipliers, obj, status = nlp.solve(x0)
time_end = time.time()
time_taken = time_end - time_start
print('   * problem successfully solved in ' + str(time_taken) + ' seconds!')
obj = -obj # NOTE: take negative here, since we solved minimization problem using negative of original maximization problem's objective function
nlp.close()

## process solution
# permanent objects created: XXX
u1 = [x[3*i] for i in range(n_type)]
u2 = [x[3*i + 1] for i in range(n_type)]
v = [x[3*i + 2] for i in range(n_type)]
v_util = disabil(type_ttheta)*v
c1 = util_inv(np.array(u1))
c2 = util_inv(np.array(u2))
s_rate = (c2/R)/(c1 + c2/R)
s_rate_lf = 1/(R*(type_bbeta*type_ddelta*R)**(-eeta) + 1)
s_rate_lf = s_rate_lf[0:n_bbeta]
s_rate_fb = 1/(R*(type_ddelta*R)**(-eeta) + 1)
s_rate_fb = s_rate_fb[0:n_bbeta]
y = disutil_inv(np.array(v))
l = y/type_ttheta
V = u1 + type_ddelta*u2 - v_util
U = u1 + type_bbeta*type_ddelta*u2 - v_util
welfare = sum(dist_types*type_pareto*V)
budget_check = sum(dist_types*(y - c1 - c2/R) - G)
budget_satisfied = (budget_check < check_tol)
ic_check = ic_comat@x
ic_satisfied = (ic_check < check_tol)
ic_satisfied_all = np.all(ic_satisfied)

## print final output
if print_final:
    print('\n*** final solution:')
    print('\n   * period 1 consumption utility (u_1) = \n ', np.around(u1, decimals = n_dec))
    print('\n   * period 2 consumption utility (u_2) = \n ', np.around(u2, decimals = n_dec))
    print('\n   * output disutility (v(y)) = \n ', np.around(v, decimals = n_dec))
    print('\n   * work effort disutility (v(l)) = \n ', np.around(v_util, decimals = n_dec))
    print('\n   * period 1 consumption (c_1) = \n ', np.around(c1, decimals = n_dec))
    print('\n   * period 1 consumption (c_2) = \n ', np.around(c2, decimals = n_dec))
    print('\n   * savings rate (s = c_2/R/(c_1 + c_2/R)) = \n ', np.around(s_rate, decimals = n_dec))
    print('\n   * laissez-faire savings rate = \n ', np.around(s_rate_lf, decimals = n_dec))
    print('\n   * first-best savings rate = \n ', np.around(s_rate_fb, decimals = n_dec))
    print('\n   * output (y) = \n ', np.around(y, decimals = n_dec))
    print('\n   * work effort (l) = \n ', np.around(l, decimals = n_dec))
    print('\n   * experienced utility (V) = \n ', np.around(V, decimals = n_dec))
    print('\n   * decision utility (U) = \n ', np.around(U, decimals = n_dec))
    print('\n   * aggregate welfare (W) = \n ', np.around(welfare, decimals = n_dec))
    print('\n   * budget (F = sum{y - c_1 - c_2/R} - G) = \n ', np.around(budget_check, decimals = n_dec))
    if budget_satisfied:
        print('   --> satisfied!')
    else:
        print('   --> violated!')
    print('\n   * maximum IC deviation = \n ', np.around(max(ic_check), decimals = n_dec))
    if ic_satisfied_all:
        print('   --> satisfied!')
    else:
        print('   --> violated!')

## test plots
# permanent objects created: 
#x_p = np.linspace(1, 10, n_type)
#plt.plot(x_p, disutil_inv(x_p))
#plt.xlabel('x')
#plt.ylabel('y')
#plt.show()

## next steps:
# - test run IPOPT problem
# - associated with each IC(i, j) one line of ic_comat, then construct the relevant constraint matrix for any arbitrary IC constraint set {(i, j)} -- solution: first create dictionary containing matrix rows, then construct constraint matrix given set of constraint numbers
# - arbitrarily loop through constraint sets, then devise active set method

## housekeeping (close)
