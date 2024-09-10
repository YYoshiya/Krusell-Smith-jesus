import numpy as np
from scipy.linalg import inv
from scipy import interpolate
from scipy.optimize import minimize_scalar
import time
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
import multiprocessing as multi
from dataclasses import dataclass, field
import quantecon as qe
from quantecon import MarkovChain
from tqdm import tqdm  # For progress bar
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

class KSSolution:
    def __init__(self, k_opt, value, B, R2):
        self.k_opt = k_opt
        self.value = value
        self.B = B
        self.R2 = R2

def KSSolution_initializer(ksp):
    # Initialize k_opt
    k_opt = ksp.beta * np.tile(ksp.k_grid[:, np.newaxis, np.newaxis], 
                               (1, ksp.K_size, ksp.s_size))
    k_opt = 0.9 * np.tile(ksp.k_grid[:, np.newaxis, np.newaxis], 
                          (1, ksp.K_size, ksp.s_size))
    k_opt = np.clip(k_opt, ksp.k_min, ksp.k_max)

    print(k_opt.shape)

    
    # Initialize value function
    value = ksp.u(0.1 / 0.9 * k_opt) / (1 - ksp.beta)
    print(value.shape)
    print(f"value max: {np.max(value)}")
    print(f"value min: {np.min(value)}")
    # Initialize B
    B = np.array([0.0, 1.0, 0.0, 1.0])

    # Create KSSolution instance
    kss = KSSolution(k_opt, value, B, [0.0, 0.0])
    return kss
@dataclass
class TransitionMatrix:
    P: np.ndarray       # 4x4
    Pz: np.ndarray      # 2x2 aggregate shock
    Peps_gg: np.ndarray # 2x2 idiosyncratic shock conditional on good to good
    Peps_bb: np.ndarray # 2x2 idiosyncratic shock conditional on bad to bad
    Peps_gb: np.ndarray # 2x2 idiosyncratic shock conditional on good to bad
    Peps_bg: np.ndarray # 2x2 idiosyncratic shock conditional on bad to good

def create_transition_matrix(ug, ub, zg_ave_dur, zb_ave_dur, ug_ave_dur, ub_ave_dur, puu_rel_gb2bb, puu_rel_bg2gg):
    # Probability of remaining in good state
    pgg = 1 - 1 / zg_ave_dur
    # Probability of remaining in bad state
    pbb = 1 - 1 / zb_ave_dur
    # Probability of changing from g to b
    pgb = 1 - pgg
    # Probability of changing from b to g
    pbg = 1 - pbb  
    
    # Probability of 0 to 0 cond. on g to g
    p00_gg = 1 - 1 / ug_ave_dur
    # Probability of 0 to 0 cond. on b to b
    p00_bb = 1 - 1 / ub_ave_dur
    # Probability of 0 to 1 cond. on g to g
    p01_gg = 1 - p00_gg
    # Probability of 0 to 1 cond. on b to b
    p01_bb = 1 - p00_bb
    
    # Probability of 0 to 0 cond. on g to b
    p00_gb = puu_rel_gb2bb * p00_bb
    # Probability of 0 to 0 cond. on b to g
    p00_bg = puu_rel_bg2gg * p00_gg
    # Probability of 0 to 1 cond. on g to b
    p01_gb = 1 - p00_gb
    # Probability of 0 to 1 cond. on b to g
    p01_bg = 1 - p00_bg
    
    # Probability of 1 to 0 cond. on g to g
    p10_gg = (ug - ug * p00_gg) / (1 - ug)
    # Probability of 1 to 0 cond. on b to b
    p10_bb = (ub - ub * p00_bb) / (1 - ub)
    # Probability of 1 to 0 cond. on g to b
    p10_gb = (ub - ug * p00_gb) / (1 - ug)
    # Probability of 1 to 0 cond. on b to g
    p10_bg = (ug - ub * p00_bg) / (1 - ub)
    # Probability of 1 to 1 cond. on g to g
    p11_gg = 1 - p10_gg
    # Probability of 1 to 1 cond. on b to b
    p11_bb = 1 - p10_bb
    # Probability of 1 to 1 cond. on g to b
    p11_gb = 1 - p10_gb
    # Probability of 1 to 1 cond. on b to g
    p11_bg = 1 - p10_bg
    
    # Constructing the transition matrices
    P = np.array([[pgg * p11_gg, pgb * p11_gb, pgg * p10_gg, pgb * p10_gb],
                  [pbg * p11_bg, pbb * p11_bb, pbg * p10_bg, pbb * p10_bb],
                  [pgg * p01_gg, pgb * p01_gb, pgg * p00_gg, pgb * p00_gb],
                  [pbg * p01_bg, pbb * p01_bb, pbg * p00_bg, pbb * p00_bb]])
    
    Pz = np.array([[pgg, pgb],
                   [pbg, pbb]])
    
    Peps_gg = np.array([[p11_gg, p10_gg],
                        [p01_gg, p00_gg]])
    
    Peps_bb = np.array([[p11_bb, p10_bb],
                        [p01_bb, p00_bb]])
    
    Peps_gb = np.array([[p11_gb, p10_gb],
                        [p01_gb, p00_gb]])
    
    Peps_bg = np.array([[p11_bg, p10_bg],
                        [p01_bg, p00_bg]])

    transmat = TransitionMatrix(P, Pz, Peps_gg, Peps_bb, Peps_gb, Peps_bg)
    return transmat

class KSParameter:
    def __init__(self, beta=0.99, alpha=0.36, delta=0.025, theta=1,
                 k_min=0.0001, k_max=1000, k_size=100, K_min=30, K_max=50, K_size=4,
                 z_min=0.99, z_max=1.01, z_size=2, eps_min=0.0, eps_max=1.0, eps_size=2,
                 ug=0.04, ub=0.1, zg_ave_dur=8, zb_ave_dur=8,
                 ug_ave_dur=1.5, ub_ave_dur=2.5, puu_rel_gb2bb=1.25, puu_rel_bg2gg=0.75,
                 mu=0, degree=7):
        
        # Utility function choice
        if theta == 1:
            self.u = self.LogUtility()
        else:
            self.u = self.CRRAUtility(theta)
        
        # Labor supply
        self.l_bar = 1 / (1 - ub)
        
        # Individual capital grid
        k_grid = (np.linspace(0, k_size - 1, k_size) / (k_size - 1))**degree * (k_max - k_min) + k_min
        k_grid[0], k_grid[-1] = k_min, k_max  # adjust numerical error
        self.k_grid = k_grid
        
        # Aggregate capital grid
        self.K_grid = np.linspace(K_min, K_max, K_size)
        
        # Aggregate technology shock
        self.z_grid = np.linspace(z_max, z_min, z_size)
        
        # Idiosyncratic employment shock grid
        self.eps_grid = np.linspace(eps_max, eps_min, eps_size)
        
        # Shock grid (s_grid)
        self.s_grid = np.array(list(itertools.product(self.z_grid, self.eps_grid)))
        
        # Transition matrices
        self.transmat = create_transition_matrix(ug, ub, zg_ave_dur, zb_ave_dur, 
                                                      ug_ave_dur, ub_ave_dur, puu_rel_gb2bb, puu_rel_bg2gg)
        
        # Other parameters
        self.beta = beta
        self.alpha = alpha
        self.delta = delta
        self.theta = theta
        self.k_min = k_min
        self.k_max = k_max
        self.K_min = K_min
        self.K_max = K_max
        self.k_size = k_size
        self.K_size = K_size
        self.z_size = z_size
        self.eps_size = eps_size
        self.s_size = z_size * eps_size
        self.ug = ug
        self.ub = ub
        self.mu = mu

    class LogUtility:
        def __call__(self, x):
            threshold = 1e-10
            x = np.asarray(x)  # x が配列でない場合でも配列に変換
            result = np.where(
                x < threshold,
                np.log(threshold) + 1e10 * (x - threshold),
                np.log(x)
            )
            return result


    class CRRAUtility:
        def __init__(self, theta):
            self.theta = theta
        
        def __call__(self, x):
            return (x**(1 - self.theta)) / (1 - self.theta)

# wage function
def w(z, K, L):
    return (1-ksp.alpha)*z*K**(ksp.alpha)*L**(-ksp.alpha)

# interest rate function
def r(z, K, L):
    return ksp.alpha*z*K**(ksp.alpha-1)*L**(1-ksp.alpha)

def compute_Kp_L(K, s_i):
    K = torch.tensor(K, dtype=torch.float32)
    if s_i % ksp.eps_size == 0:
        Kp = model(K).detach().numpy()
        L = ksp.l_bar * (1-ksp.ug)
    else:
        Kp = model(K).detach().numpy()
        L = ksp.l_bar * (1-ksp.ub)


def rhs_bellman(kp,value,k,K,s_i):
    z, eps = ksp.s_grid[s_i, 0], ksp.s_grid[s_i, 1]
    Kp, L = compute_Kp_L(K,s_i)
    c = (r(z,K,L)+1-ksp.delta)*k+w(z,K,L)*(eps*ksp.l_bar+(1.0-eps)*ksp.mu)-kp
    expec = compute_expectation(kp,Kp,value,s_i)
    return ksp.u(c)+ksp.beta*expec

def compute_expectation(kp, Kp, value, s_i):
    expec = 0
    for s_n_i in range(4):
        # `RectBivariateSpline`を使用
        value_itp = RegularGridInterpolator((ksp.k_grid, ksp.K_grid), value[:, :, s_n_i])
        expec += ksp.transmat.P[s_i, s_n_i] * value_itp((kp, Kp))
    return expec

def maximize_rhs(k_i, K_i, s_i):
    k_min, k_max = ksp.k_grid[0], ksp.k_grid[-1]
    k=ksp.k_grid[k_i]
    K=ksp.K_grid[K_i]
    z, eps = ksp.s_grid[s_i, 0], ksp.s_grid[s_i, 1]
    Kp, L = compute_Kp_L(K,s_i)
    k_c_pos = (r(z,K,L)+1-ksp.delta)*k+w(z,K,L)*(eps*ksp.l_bar+(1.0-eps)*ksp.mu)
    def obj(kp):
        return -rhs_bellman(kp, kss.value, k, K, s_i)
    
    res = minimize_scalar(obj, bounds=(k_min, min(k_c_pos, k_max)), method='bounded')
    
    # 最適化結果の取得
    kss.k_opt[k_i, K_i, s_i] = res.x
    kss.value[k_i, K_i, s_i] = -res.fun  # 最大化された値（マイナス符号を戻す）



def solve_ump(tol=1e-8, max_iter=100):
    counter_VFI = 0
    while True:
        counter_VFI += 1
        value_old = np.copy(kss.value)
        for k_i in range(ksp.k_size):
            for K_i in range(ksp.K_size):
                for s_i in range(ksp.s_size):
                    maximize_rhs(k_i, K_i, s_i)
        iterate_policy(ksp, kss, n_iter=20)
        dif = np.max(np.abs(value_old - kss.value))
        print(f"counter: {counter_VFI}, dif: {dif}")
        if dif < tol or counter_VFI >= max_iter:
            break

def iterate_policy(ksp, kss, n_iter=20):
    for _ in range(n_iter):
        # update value using policy
        value = np.array([
            rhs_bellman(kss.k_opt[k_i, K_i, s_i], kss.value,
                        ksp.k_grid[k_i], ksp.K_grid[K_i], s_i)
            for k_i in range(ksp.k_size)
            for K_i in range(ksp.K_size)
            for s_i in range(ksp.s_size)
        ]).reshape(ksp.k_size, ksp.K_size, ksp.s_size)
        
        kss.value = np.copy(value)
    
    return None


def generate_shocks(z_shock_size, population):
    mc = MarkovChain(ksp.transmat.Pz)
    zi_shock = mc.simulate(ts_length=z_shock_size)
    # idiosyncratic shocks
    epsi_shock = np.empty((z_shock_size, population), dtype=int)
    rand_draw = np.random.rand(population)
    if zi_shock[0] == 0:  # if good
        epsi_shock[0, :] = (rand_draw < ksp.ug).astype(int) + 1
    elif zi_shock[0] == 1:  # if bad
        epsi_shock[0, :] = (rand_draw < ksp.ub).astype(int) + 1
    else:
        raise ValueError(f"the value of zi_shock[0] ({zi_shock[0]}) is strange")
    
    for t in range(1, z_shock_size):
        draw_eps_shock_wrapper(zi_shock[t], zi_shock[t-1], epsi_shock[t, :], epsi_shock[t-1, :], ksp.transmat)

    for t in range(z_shock_size):
        n_e = np.sum(epsi_shock[t, :] == 1)  # Count number of employed
        empl_rate_ideal = 1.0 - ksp.ug if zi_shock[t] == 1 else 1.0 - ksp.ub
        gap = round(empl_rate_ideal * population) - n_e
        
        if gap > 0:
            # Select unemployed individuals to become employed
            unemployed_indices = np.where(epsi_shock[t, :] == 2)[0]
            if len(unemployed_indices) > 0:
                become_employed_i = np.random.choice(unemployed_indices, gap, replace=False)
                epsi_shock[t, become_employed_i] = 1
        elif gap < 0:
            # Select employed individuals to become unemployed
            employed_indices = np.where(epsi_shock[t, :] == 1)[0]
            if len(employed_indices) > 0:
                become_unemployed_i = np.random.choice(employed_indices, -gap, replace=False)
                epsi_shock[t, become_unemployed_i] = 2
    
    return zi_shock, epsi_shock

#Define the main function that does the work
def draw_eps_shock(epsi_shocks, epsi_shock_before, Peps):
    # loop over entire population
    for i in range(len(epsi_shocks)):
        rand_draw = np.random.rand()
        if epsi_shock_before[i] == 1:
            epsi_shocks[i] = 1 if Peps[0, 0] >= rand_draw else 2  # Employed before
        else:
            epsi_shocks[i] = 1 if Peps[1, 0] >= rand_draw else 2  # Unemployed before

# Wrapper function that selects the correct transition matrix
def draw_eps_shock_wrapper(zi, zi_lag, epsi_shocks, epsi_shock_before, transmat):
    if zi == 0 and zi_lag == 0:
        Peps = transmat.Peps_gg
    elif zi == 0 and zi_lag == 1:
        Peps = transmat.Peps_bg
    elif zi == 1 and zi_lag == 0:
        Peps = transmat.Peps_gb
    elif zi == 1 and zi_lag == 1:
        Peps = transmat.Peps_bb
    else:
        raise ValueError("Invalid zi or zi_lag value")
    
    # draw_eps_shock関数を呼び出し、値を引き渡す
    draw_eps_shock(epsi_shocks, epsi_shock_before, Peps)
    
class Stochastic:
    def __init__(self, zi_shocks, epsi_shocks: np.ndarray):
        self.epsi_shocks = epsi_shocks
        # `fill(40, size(epsi_shocks, 2))` 相当
        self.k_population = np.full(epsi_shocks.shape[1], 40)#これは資産の分布
        self.K_ts = np.empty(len(zi_shocks))#これはaggregate capitalというか平均
#初期値は全員がｋを40を持っているとする。


def simulate_aggregate_path(ksp, kss, zi_shocks, ss):
    epsi_shocks = ss.epsi_shocks

    T = len(zi_shocks)  # simulated duration
    N = epsi_shocks.shape[1]  # number of agents

    # Loop over T periods with progress bar
    for t, z_i in enumerate(tqdm(zi_shocks, desc="Simulating aggregate path", mininterval=0.5)):
        ss.K_ts[t] = np.mean(ss.k_population)  # current aggregate capital
        
        # Loop over individuals
        for i, k in enumerate(ss.k_population):
            eps_i = epsi_shocks[t, i]  # idiosyncratic shock
            s_i = epsi_zi_to_si(eps_i, z_i, ksp.z_size)  # transform (z_i, eps_i) to s_i
            
            # Obtain next capital holding by interpolation
            itp_pol = RegularGridInterpolator((ksp.k_grid, ksp.K_grid), kss.k_opt[:, :, s_i], bounds_error=False, fill_value=None)
            ss.k_population[i] = itp_pol((k, ss.K_ts[t]))

    return None

def epsi_zi_to_si(eps_i, z_i, z_size):
    return z_i + z_size * (eps_i - 1)


def find_ALM_coef_nn(zi_shocks, tol_ump=1e-8, max_iter_ump=100,
                  tol_B=1e-8, max_iter_B=20, T_discard=100):
    counter_B = 0
    while True:
        model.to("cpu")
        counter_B += 1
        print(f" --- Iteration over ALM coefficient: {counter_B} ---")
        
        # Solve individual problem
        solve_ump(max_iter=max_iter_ump, tol=tol_ump)
        
        # Compute aggregate path of capital
        simulate_aggregate_path(ksp, kss, zi_shocks, ss)

        model.to(device)
        ALM_nn(ksp, kss, zi_shocks, T_discard=T_discard)

        if counter_B >= max_iter_B:
            print("----------------------------------------------------------------")
            print(f"Iteration over ALM coefficient reached its maximum ({max_iter_B})")
            print("----------------------------------------------------------------")
            break

class Model(nn.Module):
    def __init__(self, d_in, d_out):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(d_in, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, d_out)
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.softplus(self.fc2(x))
        x = self.fc3(x)
        return x

class ALMDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        x = self.input_data[idx]
        y = self.output_data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def ALM_nn(ksp, kss, zi_shocks, T_discard, batch_size=32, validation_split=0.2):
    x_g = np.empty(len(zi_shocks) - T_discard - 1)
    y_g = np.empty(len(zi_shocks) - T_discard - 1)
    idx = 0
    zi_shocks_x = zi_shocks[T_discard:-1]
    zi_shocks_y = zi_shocks[T_discard + 1:]
    for t in range(T_discard, len(zi_shocks) - 1):
        x_g[idx] = ss.K_ts[t]
        y_g[idx] = ss.K_ts[t + 1]
        idx += 1

    input_data = np.column_stack((x_g, zi_shocks_x))
    output_data = y_g
    dataset = ALMDataset(input_data, output_data)
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    d_in = 2
    d_out = 1
    model = Model(d_in, d_out)
    loss_fn = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 1000
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            predictions = model(inputs).squeeze()
            loss = loss_fn(predictions, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {running_loss / len(train_loader)}')


        


ksp = KSParameter()
kss = KSSolution_initializer(ksp)
zi_shocks, epsi_shocks = generate_shocks(z_shock_size=2000, population=10000) #1100から100に変更
ss = Stochastic(zi_shocks, epsi_shocks)
T_discard = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(2, 1)
find_ALM_coef_nn(zi_shocks, 
            tol_ump = 1e-8, max_iter_ump = 10000,
            tol_B = 1e-8, max_iter_B = 3, 
            T_discard = T_discard)


def compute_approxKprime(K, z, B):
    if z == 0:
        return np.exp(B[0] + B[1] * np.log(K))
    elif z == 1:
        return np.exp(B[2] + B[3] * np.log(K))
    else:
        raise ValueError("Unexpected value of z.")

def plot_ALM(z_grid, zi_shocks, B, K_ts, T_discard=100):
    # Preallocate K_ts_approx with the same size as K_ts
    K_ts_approx = np.zeros_like(K_ts)

    # Initialize the approximate ALM with the initial value
    K_ts_approx[T_discard] = K_ts[T_discard]

    # Compute the approximate ALM for capital
    for t in range(T_discard, len(zi_shocks) - 1):
        K_ts_approx[t + 1] = compute_approxKprime(K_ts_approx[t], zi_shocks[t], B)

    # Plot the results
    plt.plot(range(T_discard + 1, len(K_ts)), K_ts[T_discard + 1:], label="true", color='red', linestyle='solid')
    plt.plot(range(T_discard + 1, len(K_ts)), K_ts_approx[T_discard + 1:], label="approximation", color='blue', linestyle='dashed')
    plt.title("Aggregate Law of Motion for Capital")
    plt.legend()
    plt.show()

def plot_Fig1(ksp, kss, K_ts):
    # K_tsの最小値と最大値を取得
    K_min, K_max = np.min(K_ts), np.max(K_ts)
    K_lim = np.linspace(K_min, K_max, 100)
    
    # Kp_gとKp_bを計算
    Kp_g = np.exp(kss.B[0] + kss.B[1] * np.log(K_lim))
    Kp_b = np.exp(kss.B[2] + kss.B[3] * np.log(K_lim))
    
    # グラフを作成
    plt.plot(K_lim, Kp_g, label="Good", linestyle='solid')
    plt.plot(K_lim, Kp_b, label="Bad", linestyle='solid')
    plt.plot(K_lim, K_lim, color='black', linestyle='dashed', label="45 degree", linewidth=0.5)
    
    plt.title("FIG1: Tomorrow's vs. today's aggregate capital")
    plt.legend()
    plt.show()


plot_ALM(ksp.z_grid, zi_shocks, kss.B, ss.K_ts, T_discard=T_discard)
plot_Fig1(ksp, kss, ss.K_ts)




    
x = ALM_nn(ksp, kss, zi_shocks=np.ones(200), T_discard=100)