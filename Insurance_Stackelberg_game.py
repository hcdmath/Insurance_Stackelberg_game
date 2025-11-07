import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# --------------------------参数设置--------------------------
params = {
    'lambda_F': 0.1,
    'lambda_L': 0.1,
    'sigma': 0.2,
    'a': 0.5,
    'theta': 1.0,  # X_F漂移项中的theta
    'rho_F': 0.05,
    'rho_L': 0.05,
    'zeta_F': 0.1,
    'zeta_L': 0.1,
    'T': 1.0,       # 终端时间
    't0': 0.0,      # 初始时间
    'x_F0': 10.0,
    'x_L0': 10.0,
    'epsilon': 0.005,    # 收敛阈值
    'max_iter': 50,     # 最大迭代次数
    # 离散化参数
    'Nt': 50,           # 时间步数
    'Nx_L': 40,         # 再保险公司状态网格数
    'Nx_F': 40,         # 保险公司状态网格数
    # 状态网格范围
    'x_L_min': 5.0,
    'x_L_max': 15.0,
    'x_F_min': 5.0,
    'x_F_max': 15.0,
    # 模拟参数
    'num_sim': 1000     # 蒙特卡洛模拟路径数
}

# --------------------------网格离散化--------------------------
# 时间网格
t_grid = np.linspace(params['t0'], params['T'], params['Nt'] + 1)
dt = t_grid[1] - t_grid[0]

# 再保险公司状态网格
x_L_grid = np.linspace(params['x_L_min'], params['x_L_max'], params['Nx_L'])
dx_L = x_L_grid[1] - x_L_grid[0]

# 保险公司状态网格
x_F_grid = np.linspace(params['x_F_min'], params['x_F_max'], params['Nx_F'])
dx_F = x_F_grid[1] - x_F_grid[0]

# --------------------------效用函数--------------------------
def U_F(x, zeta_F):
    """保险公司效用函数（指数效用）"""
    return -1 / zeta_F * np.exp(-zeta_F * x)

def U_L(x, zeta_L):
    """再保险公司效用函数（指数效用）"""
    return -1 / zeta_L * np.exp(-zeta_L * x)

# --------------------------导数计算（有限差分）--------------------------
def compute_derivatives(V, x_grid, dx):
    """
    计算价值函数的一阶导数Vx和二阶导数Vxx（中心差分）
    V: 价值函数在某一时间点的状态分布 (Nx,)
    """
    Nx = len(x_grid)
    Vx = np.zeros(Nx)
    Vxx = np.zeros(Nx)
    
    # 内部点用中心差分
    Vx[1:-1] = (V[2:] - V[:-2]) / (2 * dx)
    Vxx[1:-1] = (V[2:] - 2 * V[1:-1] + V[:-2]) / (dx ** 2)
    
    # 边界点用向前/向后差分
    Vx[0] = (V[1] - V[0]) / dx
    Vx[-1] = (V[-1] - V[-2]) / dx
    
    Vxx[0] = (V[1] - 2 * V[0] + V[0]) / (dx **2)  # 向前差分近似
    Vxx[-1] = (V[-1] - 2 * V[-2] + V[-3]) / (dx** 2)
    
    return Vx, Vxx

# --------------------------辅助函数：找到网格索引--------------------------
def find_grid_index(x, grid):
    """找到值x在网格中的最近索引"""
    return np.argmin(np.abs(grid - x))

# --------------------------初始化--------------------------
# 初始时间索引
k0 = 0  # t0对应的时间索引
# 初始状态索引
idx_L0 = find_grid_index(params['x_L0'], x_L_grid)
idx_F0 = find_grid_index(params['x_F0'], x_F_grid)

# 初始化价值函数（终端时间为效用函数，其他时间初始化为终端值）
V_L = np.zeros((params['Nt'] + 1, params['Nx_L'])) 
V_F = np.zeros((params['Nt'] + 1, params['Nx_F']))

for i in range(params['Nx_L']):
    V_L[-1, i] = U_L(x_L_grid[i], params['zeta_L'])  # 终端时间价值
for k in range(params['Nt']):
    V_L[k, :] = V_L[-1, :]  # 初始价值函数

for i in range(params['Nx_F']):
    V_F[-1, i] = U_F(x_F_grid[i], params['zeta_F'])
for k in range(params['Nt']):
    V_F[k, :] = V_F[-1, :]  # 初始价值函数

# 计算初始导数
Vx_L0, Vxx_L0 = compute_derivatives(V_L[k0, :], x_L_grid, dx_L)
Vx_F0, Vxx_F0 = compute_derivatives(V_F[k0, :], x_F_grid, dx_F)
vx_L0 = Vx_L0[idx_L0]
vxx_L0 = Vxx_L0[idx_L0]
vx_F0 = Vx_F0[idx_F0]
vxx_F0 = Vxx_F0[idx_F0]

# 计算初始A^(0,0)和B^(0,0)
sigma = params['sigma']
a_param = params['a']

A00 = (2 * vx_F0 * vxx_F0 * vx_L0 + (vx_F0 ** 2) * vxx_L0) / (sigma ** 2 * (vxx_F0 ** 2))

B00_num = (sigma**2 * (vxx_F0**2) * vx_L0 
          - 2 * a_param * vxx_F0 * vx_F0 * vx_L0 
          + sigma**2 * vx_F0 * vxx_F0 * vxx_L0 
          - a_param * (vx_F0**2) * vxx_L0)
B00 = B00_num / (sigma**2 * (vxx_F0**2))

# 初始化策略参数
lambda_L = params['lambda_L']
lambda_F = params['lambda_F']

# 初始π^(0)（指数分布参数）
discriminant = B00**2 - 4 * A00 * lambda_L
if discriminant < 0:
    raise ValueError("The discriminant of the initial $\\pi$ is negative and cannot be calculated.")
sqrt_disc = np.sqrt(discriminant)
lambda_π0 = (-B00 + sqrt_disc) / (2 * lambda_L)
if lambda_π0 <= 0:
    raise ValueError("The initial $\\pi$ parameter is non-positive and not valid.")

# 初始γ^(0)（正态分布参数：均值和方差）
v0 = (-lambda_F) / (sigma**2 * vxx_F0)
if v0 <= 0:
    raise ValueError("The initial $\\gamma$ variance is non-positive and not valid.")

m0_num = (2 * A00 * a_param + B00 + sqrt_disc) * vx_F0
m0_den = 2 * A00 * sigma**2 * vxx_F0
m0 = m0_num / m0_den

# 存储策略迭代历史，以便后续调用
pi_params = [lambda_π0]
gamma_params = [(m0, v0)]

# 存储价值函数迭代历史（初始状态点）
V_L_history = [V_L[k0, idx_L0]]
V_F_history = [V_F[k0, idx_F0]]

# --------------------------主迭代过程--------------------------
s = 0
converged = False
print("开始迭代...")

while s < params['max_iter'] and not converged:
    # 步骤2：再保险公司（L）更新
    lambda_π_s = pi_params[s]
    m_s, v_s = gamma_params[s]
    
    # 1. 价值函数V^L(s+1)更新：模拟X_L路径计算期望
    num_sim = params['num_sim']
    X_L_paths = np.zeros((num_sim, params['Nt'] + 1))
    X_L_paths[:, 0] = params['x_L0']  # 初始状态
    
    for k in range(params['Nt']):
        # 漂移项：ρ_L X_L + (1 - m_s)(1/λ_π_s - a)
        drift = params['rho_L'] * X_L_paths[:, k] + (1 - m_s) * (1 / lambda_π_s - a_param)
        # 扩散项：sqrt((1 - m_s)^2 + v_s) * σ * dW
        diffusion_coeff = np.sqrt((1 - m_s)**2 + v_s) * sigma
        dW = np.random.normal(0, np.sqrt(dt), size=num_sim)
        X_L_paths[:, k+1] = X_L_paths[:, k] + drift * dt + diffusion_coeff * dW
    
    # 计算价值函数更新值（积分项+终端效用）
    integral_term_L = lambda_L * (params['T'] - params['t0']) * (1 - np.log(lambda_π_s))
    terminal_utility_L = U_L(X_L_paths[:, -1], params['zeta_L'])
    V_L_new = V_L.copy()
    V_L_new[k0, idx_L0] = np.mean(integral_term_L + terminal_utility_L)
    
    # 2. 策略π^(s+1)更新
    Vx_L_new, Vxx_L_new = compute_derivatives(V_L_new[k0, :], x_L_grid, dx_L)
    vx_L_s1 = Vx_L_new[idx_L0]
    vxx_L_s1 = Vxx_L_new[idx_L0]
    
    Vx_F_s, Vxx_F_s = compute_derivatives(V_F[k0, :], x_F_grid, dx_F)
    vx_F_s = Vx_F_s[idx_F0]
    vxx_F_s = Vxx_F_s[idx_F0]
    
    # 计算A^(s+1,s)和B^(s+1,s)
    A_s1s = (2 * vx_F_s * vxx_F_s * vx_L_s1 + (vx_F_s **2) * vxx_L_s1) / (sigma** 2 * (vxx_F_s **2))
    B_s1s_num = (sigma**2 * (vxx_F_s**2) * vx_L_s1 
                - 2 * a_param * vxx_F_s * vx_F_s * vx_L_s1 
                + sigma**2 * vx_F_s * vxx_F_s * vxx_L_s1 
                - a_param * (vx_F_s**2) * vxx_L_s1)
    B_s1s = B_s1s_num / (sigma**2 * (vxx_F_s**2))
    
    # 更新π参数
    discriminant_pi = B_s1s**2 - 4 * A_s1s * lambda_L
    if discriminant_pi < 0: #指数分布参数必须大于0
        print(f"Iteration{s+1}: The discriminant of $\\pi$ is negative, so stop the iteration.")
        break
    sqrt_disc_pi = np.sqrt(discriminant_pi)
    lambda_π_s1 = (-B_s1s + sqrt_disc_pi) / (2 * lambda_L)
    if lambda_π_s1 <= 0:
        print(f"Iteration {s+1}: The $\\pi$ parameter is non-positive, so stop the iteration.")
        break
    pi_params.append(lambda_π_s1)
    
    # 步骤3：保险公司（F）更新
    # 1. 价值函数V^F(s+1)更新：模拟X_F路径计算期望
    X_F_paths = np.zeros((num_sim, params['Nt'] + 1))
    X_F_paths[:, 0] = params['x_F0']
    
    for k in range(params['Nt']):
        # 漂移项：ρ_F X_F + θa - (1 - m_s)(1/λ_π_s1 - a)
        drift_F = (params['rho_F'] * X_F_paths[:, k] 
                  + params['theta'] * a_param 
                  - (1 - m_s) * (1 / lambda_π_s1 - a_param))
        # 扩散项：sqrt(m_s² + v_s) * σ * dW
        diffusion_coeff_F = np.sqrt(m_s**2 + v_s) * sigma
        dW = np.random.normal(0, np.sqrt(dt), size=num_sim)
        X_F_paths[:, k+1] = X_F_paths[:, k] + drift_F * dt + diffusion_coeff_F * dW
    
    # 计算价值函数更新值
    log_arg = (-2 * lambda_F * lambda_π_s1 * np.e) / (sigma**2 * vxx_F_s)
    if log_arg <= 0:#正太分布方差是一个必须大于0的数
        print(f"Iteration {s+1}: The logarithmic parameter of F is non-positive, so stop the iteration.")
        break
    integral_term_F = (lambda_F / 2) * np.log(log_arg) * (params['T'] - params['t0'])
    terminal_utility_F = U_F(X_F_paths[:, -1], params['zeta_F'])
    V_F_new = V_F.copy()
    V_F_new[k0, idx_F0] = np.mean(integral_term_F + terminal_utility_F)
    
    # 2. 策略γ^(s+1)更新
    Vx_F_s1, Vxx_F_s1 = compute_derivatives(V_F_new[k0, :], x_F_grid, dx_F)
    vx_F_s1 = Vx_F_s1[idx_F0]
    vxx_F_s1 = Vxx_F_s1[idx_F0]
    
    # 计算A^(s+1,s+1)和B^(s+1,s+1)
    A_s1s1 = (2 * vx_F_s1 * vxx_F_s1 * vx_L_s1 + (vx_F_s1 **2) * vxx_L_s1) / (sigma** 2 * (vxx_F_s1 **2))
    B_s1s1_num = (sigma**2 * (vxx_F_s1**2) * vx_L_s1 
                 - 2 * a_param * vxx_F_s1 * vx_F_s1 * vx_L_s1 
                 + sigma**2 * vx_F_s1 * vxx_F_s1 * vxx_L_s1 
                 - a_param * (vx_F_s1**2) * vxx_L_s1)
    B_s1s1 = B_s1s1_num / (sigma**2 * (vxx_F_s1**2))
    
    # 更新γ参数
    v_s1 = (-lambda_F) / (sigma**2 * vxx_F_s1)
    if v_s1 <= 0:
        print(f"Iteration {s+1}: The $\\gamma$ variance is non-positive, so stop the iteration.")
        break
    
    discriminant_gamma = B_s1s1**2 - 4 * A_s1s1 * lambda_L
    if discriminant_gamma < 0:
        print(f"Iteration {s+1}: The discriminant of the $\\gamma$ mean is negative, so stop the iteration.")
        break
    sqrt_disc_gamma = np.sqrt(discriminant_gamma)
    m_s1_num = (2 * A_s1s1 * a_param + B_s1s1 + sqrt_disc_gamma) * vx_F_s1
    m_s1_den = 2 * A_s1s1 * sigma**2 * vxx_F_s1
    m_s1 = m_s1_num / m_s1_den
    gamma_params.append((m_s1, v_s1))
    
    # 记录价值函数历史
    V_L_history.append(V_L_new[k0, idx_L0])
    V_F_history.append(V_F_new[k0, idx_F0])
    
    # 收敛判断
    diff_L = np.abs(V_L_new[k0, idx_L0] - V_L[k0, idx_L0])
    diff_F = np.abs(V_F_new[k0, idx_F0] - V_F[k0, idx_F0])
    print(f"Iteration {s+1}: V_L change = {diff_L:.6f}, V_F change = {diff_F:.6f}")
    
    if diff_L <= params['epsilon'] and diff_F <= params['epsilon']:
        converged = True
        V_L = V_L_new
        V_F = V_F_new
        break
    
    # 更新价值函数
    V_L = V_L_new
    V_F = V_F_new
    s += 1

# --------------------------结果输出--------------------------
if converged:
    print("\n===== Convergence results =====")
    print(f"optimal π(parameter of exponential distribution):{pi_params[-1]:.6f}")
    print(f"Optimal γ(normal distribution): mean={gamma_params[-1][0]:.6f}, variance={gamma_params[-1][1]:.6f}")
    print(f"Reinsurer's value function:V_L*={V_L[k0, idx_L0]:.6f}")
    print(f"Insurer's value function:V_F*={V_F[k0, idx_F0]:.6f}")
else:
    print("\n===== Not convergent =====")
    print(f"The maximum number of iterations ({params['max_iter']}) has been reached, and the current result is the value of the last iteration.")

# --------------------------保险人和再保险人的值函数图像--------------------------
# 选择需要展示的时间点（时间索引）
time_indices = [params['Nt']//2]
time_labels = [f"t={t_grid[k]:.2f}" for k in time_indices]

# 1. 再保险人的值函数
fig_reinsurer_time, ax_re = plt.subplots(figsize=(8, 5))
for k, label in zip(time_indices, time_labels):
    ax_re.plot(x_L_grid, V_L[k, :], linewidth=3, label="$V^L(t, x_L)$",color="#ff0000",linestyle='-')
ax_re.set_xlabel('Surplus $x_L$', fontsize=17)
ax_re.set_ylabel('Value function', fontsize=17)
ax_re.tick_params(axis='both', which='major', labelsize=14)  # 主要刻度
ax_re.tick_params(axis='both', which='minor', labelsize=14)  # 次要刻度
plt.legend(fontsize=20, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 2. 保险人的值函数
fig_insurer_time, ax_in = plt.subplots(figsize=(8, 5))
for k, label in zip(time_indices, time_labels):
    ax_in.plot(x_F_grid, V_F[k, :], linewidth=3, label="$V^F(t, x_F)$", color="#0000FF",linestyle='--')
ax_in.set_xlabel('Surplus $x_F$', fontsize=17)
ax_in.set_ylabel('Value function', fontsize=17)
ax_in.tick_params(axis='both', which='major', labelsize=14)  # 主要刻度
ax_in.tick_params(axis='both', which='minor', labelsize=14)  # 次要刻度
plt.legend(fontsize=20, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --------------------------最优策略（Π和γ）的概率密度函数绘图--------------------------
def exponential_pdf(x, lam):
    """指数分布概率密度函数（x≥0）"""
    return lam * np.exp(-lam * x) if x >= 0 else 0

def normal_pdf(x, mu, var):
    """正态分布概率密度函数"""
    sigma = np.sqrt(var)
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x - mu)**2 / (2 * var))

# 提取最优策略参数
opt_lam_π = pi_params[-1]  # 最优Π的指数分布率参数
opt_mu_γ, opt_var_γ = gamma_params[-1]  # 最优γ的正态分布均值和方差
opt_sigma_γ = np.sqrt(opt_var_γ)

x_π = np.linspace(0, -np.log(0.01) / opt_lam_π, 200)
pdf_π = [exponential_pdf(x, opt_lam_π) for x in x_π]

x_γ = np.linspace(opt_mu_γ - 3 * opt_sigma_γ, opt_mu_γ + 3 * opt_sigma_γ, 200)
pdf_γ = [normal_pdf(x, opt_mu_γ, opt_var_γ) for x in x_γ]

# 绘制最优策略的PDF图
fig_opt_policy, (ax_opt_π, ax_opt_γ) = plt.subplots(1, 2, figsize=(16, 5))

# 1. 最优Π策略（指数分布）
ax_opt_π.plot(x_π, pdf_π, linestyle='-', linewidth=3,color='red')
# 标注均值（指数分布均值=1/λ）
mean_π = 1 / opt_lam_π
ax_opt_π.axvline(x=mean_π, color='g', linestyle='--', linewidth=3, 
                 label=f'μ= {mean_π:.4f}')
ax_opt_π.set_xlabel('p', fontsize=17)
ax_opt_π.set_ylabel('Probability density', fontsize=17)
ax_opt_π.tick_params(axis='both', which='major', labelsize=14)  # 主要刻度
ax_opt_π.tick_params(axis='both', which='minor', labelsize=14)  # 次要刻度
ax_opt_π.legend(fontsize=20)
ax_opt_π.grid(True, linestyle='--', alpha=0.7)

# 2. 最优γ策略（正态分布）
ax_opt_γ.plot(x_γ, pdf_γ, linestyle='-', linewidth=3, color='blue')
ax_opt_γ.axvline(x=opt_mu_γ, color='g', linestyle='--', linewidth=3, 
                 label=f' μ = {opt_mu_γ:.6f}')
ax_opt_γ.axvline(x=opt_mu_γ - opt_sigma_γ, color='orange', linestyle=':', linewidth=2, 
                 label=f'μ-σ = {opt_mu_γ - opt_sigma_γ:.6f}')
ax_opt_γ.axvline(x=opt_mu_γ + opt_sigma_γ, color='orange', linestyle=':', linewidth=2, 
                 label=f'μ+σ = {opt_mu_γ + opt_sigma_γ:.6f}')
ax_opt_γ.set_xlabel('q', fontsize=17)
ax_opt_γ.set_ylabel('Probability density', fontsize=17)
ax_opt_γ.tick_params(axis='both', which='major', labelsize=14)  # 主要刻度
ax_opt_γ.tick_params(axis='both', which='minor', labelsize=14)  # 次要刻度
plt.legend(fontsize=18)
ax_opt_γ.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# --------------------------迭代收敛情况--------------------------
# 2. 价值函数迭代收敛曲线
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(np.arange(len(V_L_history)), V_L_history, 'r-', marker='x', label='Reinsurer $V^L$', linewidth=3)
ax3.plot(np.arange(len(V_F_history)), V_F_history, 'b--', marker='o', label='Insurer $V^F$', linewidth=3)
ax3.set_xlabel('Iterations', fontsize=17)
ax3.set_ylabel('Value function (initial state point)', fontsize=17)
ax3.tick_params(axis='both', which='major', labelsize=14)  # 主要刻度
ax3.tick_params(axis='both', which='minor', labelsize=14)  # 次要刻度
ax3.legend(fontsize=20)
ax3.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# 3. 状态过程模拟路径（最后一次迭代的策略）
if 'X_L_paths' in locals() and 'X_F_paths' in locals():
    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    # 再保险公司状态路径（随机选10条路径）
    sample_idx = np.random.choice(params['num_sim'], 10, replace=False)
    ax4a.plot(t_grid, X_L_paths[sample_idx].T, alpha=0.9)
    ax4a.set_ylabel('Reinsurer’s surplus $X_L(t)$', fontsize=14)
    ax4a.tick_params(axis='both', which='major', labelsize=10)  # 主要刻度
    ax4a.tick_params(axis='both', which='minor', labelsize=10)  # 次要刻度
    ax4a.grid(True, linestyle='--', alpha=0.7)
    
    # 保险公司状态路径
    ax4b.plot(t_grid, X_F_paths[sample_idx].T, alpha=0.9)
    ax4b.set_xlabel('Time $t$', fontsize=14)
    ax4b.set_ylabel('Insurer’s surplus $X_F(t)$', fontsize=14)
    ax4b.tick_params(axis='both', which='major', labelsize=10)  # 主要刻度
    ax4b.tick_params(axis='both', which='minor', labelsize=10)  # 次要刻度
    ax4b.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

plt.show()