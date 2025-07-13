import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# フォント設定
plt.rcParams['font.family'] = 'MS Gothic'

# ─── モデルパラメータ ─────────────────────────
d_nozzle  = 0.005                    # m, ノズル径
A_nozzle  = np.pi*(d_nozzle/2)**2    # m^2, ノズル断面積
l_nozzle  = 0.5                      # m, ノズル相当長さ
V_pipe    = A_nozzle * l_nozzle      # m^3, パイプ体積
M_boat    = 0.023                    # kg, 船質量
k_drag    = 0.001                    # N·s/m, 線形抵抗係数
Q         = (np.pi*0.4**2*4.5*0.9*50000/1140)*3  # W, 加熱入力
h_fg      = 2260e3                  # J/kg, 蒸発潜熱
R         = 461.5                   # J/(kg·K), 水蒸気の気体定数
T         = 373.15                  # K, 飽和蒸気温度
P_atm     = 101325.0                # Pa, 大気圧
rho       = 1000.0                  # kg/m^3, 水の密度

# ─── ODE 系の定義 ────────────────────────────
# 状態 y = [m, v, x]
def poppop_ode_simple(t, y):
    m, v, x = y

    # 1) 管内圧力 P_int
    P_int = (m * R * T) / V_pipe

    # 2) 噴出速度 v_exit（理想ベルヌーイ）
    delta_P = max(P_int - P_atm, 0.0)
    v_exit = np.sqrt(2 * delta_P / rho)

    # 3) 質量流出率 dot_m_out（Cd=1）
    dot_m_out = rho * A_nozzle * v_exit

    # 4) 質量収支 dm/dt
    dm_dt = Q / h_fg - dot_m_out

    # 5) 推力 F
    F = dot_m_out * v_exit

    # 6) 抵抗力を「v>0 のときのみ」適用し，速度逆転防止
    drag = k_drag * v if v > 0 else 0.0
    dv_dt = (F - drag) / M_boat
    if v <= 0 and dv_dt < 0:
        dv_dt = 0.0

    # 7) 位置変化 dx/dt
    dx_dt = v

    return [dm_dt, dv_dt, dx_dt]

# ─── シミュレーション実行 ─────────────────────
y0     = [0.0, 0.0, 0.0]    # 初期値: m=0, v=0, x=0
t_span = (0.0, 20.0)       # 0～20秒
sol    = solve_ivp(poppop_ode_simple, t_span, y0, max_step=0.01)

# ─── 結果プロット ───────────────────────────
t = sol.t
m, v, x = sol.y

plt.figure(figsize=(8,5))
plt.plot(t, v, label='速度 $v(t)$ [m/s]')
plt.plot(t, x, '--', label='位置 $x(t)$ [m]')
plt.xlabel('時間 [s]')
plt.grid(True)
plt.legend()
plt.title('Cd 無視（理想ノズル）でのポンポン船モデル（速度逆転防止）')
plt.tight_layout()
plt.show()
