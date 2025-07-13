import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'

# ─── モデルパラメータ ─────────────────────────
d_nozzle = 0.005                       # m, ノズル径
A_nozzle = np.pi * (d_nozzle / 2) ** 2 # m^2, ノズル断面積
V_pipe   = A_nozzle * 0.0125           # m^3，パイプ中の蒸気体積
M_boat   = 0.023                       # kg, 船質量
C_d_boat = 0.00001                       # 船の抵抗係数 
A_front  = 0.01*0.1                       # 船の正面面積 [m^2]
Q        = ((np.pi * (0.4**2) * 4.5 * 0.9 * 46000) / 1140) * 3  # W, 加熱入力
h_fg     = 2260e3                      # J/kg, 蒸発潜熱
R        = 461.5                       # J/(kg·K), 水蒸気の気体定数
T        = 373.15                      # K, 飽和蒸気温度
P_atm    = 101325.0                    # Pa, 大気圧
rho      = 1000.0                      # kg/m^3, 水の密度

# ─── シミュレーション設定 ────────────────────────
dt    = 0.001    # s, タイムステップ
t_end = 20.0     # s, シミュレーション終了時間
N     = int(t_end / dt) + 1
t     = np.linspace(0, t_end, N)

# 状態変数と圧力配列の初期化
m = np.zeros(N)   # パイプ内蒸気質量
v = np.zeros(N)   # 船速度
x = np.zeros(N)   # 船位置
P = np.zeros(N)   # 管内圧力

# 初期条件
m[0] = 0.0
v[0] = 0.0
x[0] = 0.0
P[0] = P_atm

# 微分関数（質量・速度のみ）
def deriv(mi, vi):
    # 現在の圧力
    P_int = (mi * R * T) / V_pipe
    # 噴出速度
    delta_P = max(P_int - P_atm, 0.0)
    v_exit = np.sqrt(2 * delta_P / rho)
    # 質量流出率
    dot_m_out = rho * A_nozzle * v_exit
    # 質量収支
    dm_dt = Q / h_fg - dot_m_out
    # 推力
    F = dot_m_out * v_exit
    # 抵抗力 (v>0 のときのみ)
    drag    = 0.5 * rho * C_d_boat * A_front * vi**2 if vi > 0 else 0.0
    # 加速度
    dv_dt = (F - drag) / M_boat
    # 速度逆転防止
    if vi <= 0 and dv_dt < 0:
        dv_dt = 0.0
    return dm_dt, dv_dt

# RK4 で積分
for i in range(N - 1):
    mi, vi = m[i], v[i]
    # 圧力記録
    P[i] = (mi * R * T) / V_pipe
    # RK4 ステップ
    k1_m, k1_v = deriv(mi, vi)
    k2_m, k2_v = deriv(mi + 0.5 * dt * k1_m, vi + 0.5 * dt * k1_v)
    k3_m, k3_v = deriv(mi + 0.5 * dt * k2_m, vi + 0.5 * dt * k2_v)
    k4_m, k4_v = deriv(mi + dt * k3_m, vi + dt * k3_v)

    m_next = mi + (dt / 6) * (k1_m + 2 * k2_m + 2 * k3_m + k4_m)
    v_next = vi + (dt / 6) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    # 速度クランプ
    if v_next < 0:
        v_next = 0.0
    x_next = x[i] + v_next * dt

    m[i + 1] = m_next
    v[i + 1] = v_next
    x[i + 1] = x_next

# 最終ステップの圧力
P[-1] = (m[-1] * R * T) / V_pipe

# ─── 結果プロット ───────────────────────────
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
axs[0].plot(t, v, label='速度 v(t) [m/s]')
axs[0].set_ylabel('速度 [m/s]')
axs[0].grid(True)
axs[0].legend()

axs[1].plot(t, x, '--', label='位置 x(t) [m]')
axs[1].set_ylabel('位置 [m]')
axs[1].grid(True)
axs[1].legend()

axs[2].plot(t, P / 1e5, color='tab:orange', label='管内圧力 P(t) [bar]')
axs[2].set_xlabel('時間 [s]')
axs[2].set_ylabel('圧力 [bar]')
axs[2].grid(True)
axs[2].legend()

fig.suptitle('ポンポン船モデル: 速度・位置・圧力の時間変化')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()