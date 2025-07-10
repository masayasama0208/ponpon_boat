# full_poppop.py  –  フル状態4変数モデル (x,v,N,U) + optional s
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------------- parameters --------------------------
P = dict(                     # ← 実験値で置き換えればOK
    At   = np.pi*(0.0025)**2,     # パイプ断面 [m²] (φ2 mm)
    L    = 0.05,                 # パイプ長 [m]
    V0   = 5.0e-7,               # ボイラー容積 [m³] = 0.5 cm³
    rho  = 1.0e3,                # 水密度 [kg/m³]
    M    = 2.0e-2,               # 船質量 [kg]
    CD   = 1.0e-2,  Aw=4.0e-4,   # 抵抗係数・投影面積
    lam  = 2.3e-3,               # 摩擦係数 λ [kg m⁻¹]
    alpha= 1.0e-6,               # 凝縮係数 α [mol Pa⁻¹ s⁻¹]
    Qdot = 0.6,    Lv = 4.07e4,  # 熱流率 [W]・潜熱 [J/mol]
    R    = 8.314,  Tw = 373.0,   # 気体定数・壁温 [K]
    p0   = 1.013e5,              # 外気圧 [Pa]
)P["mw"]  = P["rho"]*P["At"]*P["L"]  # パイプ内水質量 [kg]
P["psat"]= P["p0"]                   # 飽和圧：今回は壁100 °C ≈ 1 atm

# -------------------------- RHS of ODE --------------------------
def rhs(t, y):
    x, v, N, U, *rest = y          # (*rest==[s] でも動く)
    V    = P["V0"] - P["At"]*x
    p_in = N*P["R"]*P["Tw"] / V
    dx   = v
    dv   = (P["At"]/P["mw"])*(p_in-P["p0"]) - P["lam"]/P["mw"]*v*abs(v)
    dN   = P["Qdot"]/P["Lv"] - P["alpha"]*(p_in - P["psat"])
    dU   = (P["mw"]/P["M"])*dv - 0.5*P["rho"]*P["CD"]*P["Aw"]*U*abs(U)/P["M"]
    ds   = U                       # 位置はお好みで
    return [dx, dv, dN, dU, ds][:len(y)]

# -------------------------- integrate ---------------------------
y0 = [0.0, 0.0, P["p0"]*P["V0"]/(P["R"]*P["Tw"]), 0.0, 0.0]  # x,v,N,U,s
sol = solve_ivp(rhs, (0, 20), y0, method='RK45', rtol=1e-6, atol=1e-9,
                max_step=1e-3, t_eval=np.linspace(0, 20, 5000))

# -------------------------- quick plot --------------------------
t = sol.t
x_mm = sol.y[0]*1e3
U_cm = sol.y[3]*1e2
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(t, x_mm); ax[0].set_ylabel('x  [mm]')
ax[1].plot(t, U_cm); ax[1].set_ylabel('U  [cm/s]')
ax[1].set_xlabel('time [s]'); fig.tight_layout(); plt.show()

# 圧力の時間変化を計算
V = P["V0"] - P["At"]*sol.y[0]
p_in = sol.y[2]*P["R"]*P["Tw"] / V

plt.figure()
plt.plot(t, p_in/1e5)  # [Pa]→[気圧]にしたい場合は/1e5
plt.xlabel('time [s]')
plt.ylabel('Pressure [atm]')
plt.title('ボイラー内圧力の時間変化')
plt.tight_layout()
plt.show()
