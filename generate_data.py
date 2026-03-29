"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  generate_data.py — CV Quantum Information Dashboard                        ║
║  Run ONCE on HPC / local machine with QuTiP + SF + PennyLane installed.     ║
║  Produces: data/states.pkl, data/channels.pkl, data/gbs.pkl                 ║
║  Then deploy app.py (zero heavy deps at runtime) to Streamlit Cloud.        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    pip install qutip numpy scipy strawberryfields thewalrus pennylane
    python generate_data.py
    # → creates data/ folder with .pkl files
    # Commit data/ to GitHub alongside app.py
"""

import math, pickle, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path

# ── QuTiP ────────────────────────────────────────────────────────────────────
import qutip as qt
from qutip import (
    basis, ket2dm, coherent, coherent_dm, thermal_dm,
    expect, destroy, num, displace, squeeze, tensor, qeye,
    mesolve, Options,
)

OUT = Path("data")
OUT.mkdir(exist_ok=True)

XVEC   = np.linspace(-6, 6, 120)   # phase-space grid (120×120)
DIM    = 40                          # Fock truncation
DIM_SM = 25                          # for small states

def make_qobj(arr):
    d = arr.shape[0]
    rho = qt.Qobj(arr)
    rho.dims = [[d], [d]]
    return rho

def wigner_arr(rho_arr):
    rho = make_qobj(rho_arr)
    return np.array(qt.wigner(rho, XVEC, XVEC, g=2))

def husimi_arr(rho_arr):
    rho = make_qobj(rho_arr)
    res = qt.qfunc(rho, XVEC, XVEC)
    return np.array(res[0] if isinstance(res, tuple) else res)

def metrics(rho_arr):
    rho  = make_qobj(rho_arr)
    dim  = rho.shape[0]
    a    = destroy(dim)
    n_op = num(dim)
    x_op = (a + a.dag()) / np.sqrt(2)
    p_op = 1j*(a.dag() - a) / np.sqrt(2)
    mn   = float(expect(n_op, rho).real)
    mn2  = float(expect(n_op*n_op, rho).real)
    vn   = mn2 - mn**2
    pur  = float((rho*rho).tr().real)
    ent  = float(qt.entropy_vn(rho, base=2))
    mq   = round((vn-mn)/mn, 5) if mn > 1e-10 else None
    mx   = float(expect(x_op, rho).real)
    mp   = float(expect(p_op, rho).real)
    vx   = float(expect(x_op*x_op, rho).real) - mx**2
    vp   = float(expect(p_op*p_op, rho).real) - mp**2
    dx   = math.sqrt(max(vx, 0.0))
    dp   = math.sqrt(max(vp, 0.0))
    probs = np.array([float(rho[n,n].real) for n in range(min(dim,30))])
    return dict(
        mean_n=round(mn,5), var_n=round(vn,5), purity=round(pur,6),
        entropy=round(ent,6), mandel_Q=mq,
        mean_x=round(mx,5), mean_p=round(mp,5),
        delta_x=round(dx,6), delta_p=round(dp,6),
        heis_prod=round(dx*dp,6), probs=probs.tolist(),
    )

def wigner_neg_vol(W):
    dx = XVEC[1]-XVEC[0]
    return float(np.sum(np.abs(W))*dx**2 - 1.0)

print("=== Generating STATE data ===")

states_data = {}

# ── 1. Fock states ─────────────────────────────────────────────────────────
print("  Fock states...")
fock_states = {}
for n in [0, 1, 2, 3, 5]:
    rho = ket2dm(basis(DIM, n)).full()
    W   = wigner_arr(rho)
    Q   = husimi_arr(rho)
    fock_states[n] = dict(rho=np.real(rho[:20,:20]).tolist(), W=W.tolist(), Q=Q.tolist(),
                           metrics=metrics(rho), wnv=wigner_neg_vol(W))
states_data['fock'] = fock_states

# ── 2. Coherent states ─────────────────────────────────────────────────────
print("  Coherent states...")
coherent_states = {}
for alpha_re, alpha_im in [(0,0),(1,0),(2,0),(1,1),(2,2),(-2,0),(0,2)]:
    rho = coherent_dm(DIM, alpha_re + 1j*alpha_im).full()
    W   = wigner_arr(rho)
    Q   = husimi_arr(rho)
    key = f"{alpha_re},{alpha_im}"
    coherent_states[key] = dict(rho=np.real(rho[:20,:20]).tolist(), W=W.tolist(), Q=Q.tolist(),
                                 metrics=metrics(rho), wnv=wigner_neg_vol(W))
states_data['coherent'] = coherent_states

# ── 3. Squeezed states ─────────────────────────────────────────────────────
print("  Squeezed states...")
squeezed_states = {}
for r, phi in [(0.5,0),(1.0,0),(1.5,0),(2.0,0),(1.0,1.5708)]:
    xi  = r*np.exp(1j*phi)
    psi = squeeze(DIM, xi)*basis(DIM, 0)
    rho = ket2dm(psi).full()
    W   = wigner_arr(rho)
    Q   = husimi_arr(rho)
    key = f"{r},{round(phi,2)}"
    squeezed_states[key] = dict(W=W.tolist(), Q=Q.tolist(), metrics=metrics(rho), wnv=wigner_neg_vol(W))
states_data['squeezed'] = squeezed_states

# ── 4. Thermal states ─────────────────────────────────────────────────────
print("  Thermal states...")
thermal_states = {}
for nbar in [0.5, 1, 2, 5, 10]:
    rho = thermal_dm(DIM, nbar).full()
    W   = wigner_arr(rho)
    Q   = husimi_arr(rho)
    thermal_states[nbar] = dict(W=W.tolist(), Q=Q.tolist(), metrics=metrics(rho), wnv=wigner_neg_vol(W))
states_data['thermal'] = thermal_states

# ── 5. Cat states ──────────────────────────────────────────────────────────
print("  Cat / NOON states...")
cat_states = {}
for alpha, sign, label in [(1.5,1,"even"),(1.5,-1,"odd"),(2.0,1,"even_2"),(2.0,-1,"odd_2")]:
    psi = (coherent(DIM, alpha) + sign*coherent(DIM, -alpha)).unit()
    rho = ket2dm(psi).full()
    W   = wigner_arr(rho)
    Q   = husimi_arr(rho)
    cat_states[label] = dict(W=W.tolist(), Q=Q.tolist(), metrics=metrics(rho), wnv=wigner_neg_vol(W),
                              alpha=alpha, sign=sign)
states_data['cat'] = cat_states

# ── 6. Displaced-squeezed ──────────────────────────────────────────────────
print("  Displaced-squeezed states...")
ds_states = {}
for alpha, r, phi in [(1+1j,0.5,0),(2+0j,1.0,0),(1+2j,0.8,1.5708)]:
    xi  = r*np.exp(1j*phi)
    psi = displace(DIM, alpha)*squeeze(DIM, xi)*basis(DIM, 0)
    rho = ket2dm(psi).full()
    W   = wigner_arr(rho)
    Q   = husimi_arr(rho)
    key = f"{alpha},{r},{round(phi,2)}"
    ds_states[key] = dict(W=W.tolist(), Q=Q.tolist(), metrics=metrics(rho), wnv=wigner_neg_vol(W))
states_data['displaced_squeezed'] = ds_states

# ── 7. GKP states ─────────────────────────────────────────────────────────
print("  GKP states...")
gkp_states = {}
for delta, n_peaks in [(0.3,3),(0.5,3)]:
    psi = sum(
        np.exp(-delta**2*n**2) * displace(DIM, n*np.sqrt(np.pi)) * basis(DIM, 0)
        for n in range(-n_peaks, n_peaks+1)
    )
    rho = ket2dm(psi.unit()).full()
    W   = wigner_arr(rho)
    Q   = husimi_arr(rho)
    key = f"{delta},{n_peaks}"
    gkp_states[key] = dict(W=W.tolist(), Q=Q.tolist(), metrics=metrics(rho), wnv=wigner_neg_vol(W))
states_data['gkp'] = gkp_states

# ── Save state data ────────────────────────────────────────────────────────
states_data['xvec'] = XVEC.tolist()
with open(OUT/"states.pkl","wb") as f:
    pickle.dump(states_data, f, protocol=4)
print(f"  ✅  data/states.pkl  ({(OUT/'states.pkl').stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Generating CHANNEL data ===")

channels_data = {}

# Base state: coherent |2⟩
rho_base  = coherent_dm(DIM, 2.0).full()

def apply_displacement(rho_arr, alpha):
    rho = make_qobj(rho_arr)
    D   = displace(rho.shape[0], alpha)
    return (D*rho*D.dag()).full()

def apply_squeeze_op(rho_arr, r):
    rho = make_qobj(rho_arr)
    S   = squeeze(rho.shape[0], r)
    return (S*rho*S.dag()).full()

def apply_phase_shift(rho_arr, phi):
    rho  = make_qobj(rho_arr)
    n_op = num(rho.shape[0])
    R    = (-1j*phi*n_op).expm()
    return (R*rho*R.dag()).full()

def apply_loss(rho_arr, gamma_t):
    rho = make_qobj(rho_arr)
    dim = rho.shape[0]
    a   = destroy(dim)
    H   = qt.Qobj(np.zeros((dim,dim))); H.dims = [[dim],[dim]]
    c_ops = [np.sqrt(1.0)*a]
    try:
        opts = Options(nsteps=15000)
        tlist = np.linspace(0, gamma_t, max(12, int(20*gamma_t)))
        res   = mesolve(H, rho, tlist, c_ops, [], options=opts)
        return res.states[-1].full()
    except Exception:
        eta = math.exp(-gamma_t); m_idx = np.arange(dim)
        decay = np.outer(np.sqrt(eta**m_idx), np.sqrt(eta**m_idx))
        return (rho.full()*decay).real

# Displacement sweep
print("  Displacement sweep...")
disp_series = []
for alpha_re in np.linspace(-3, 3, 9):
    rho = apply_displacement(rho_base, alpha_re)
    W   = wigner_arr(rho)
    disp_series.append(dict(alpha_re=float(alpha_re), W=W.tolist(), metrics=metrics(rho)))
channels_data['displacement_sweep'] = disp_series

# Squeezing sweep
print("  Squeezing sweep...")
sq_series = []
for r in [0.0, 0.3, 0.6, 1.0, 1.5, 2.0]:
    rho = apply_squeeze_op(rho_base, r)
    W   = wigner_arr(rho)
    sq_series.append(dict(r=float(r), W=W.tolist(), metrics=metrics(rho)))
channels_data['squeezing_sweep'] = sq_series

# Phase shift sweep
print("  Phase shift sweep...")
ph_series = []
for phi in np.linspace(0, 2*np.pi, 9):
    rho = apply_phase_shift(rho_base, phi)
    W   = wigner_arr(rho)
    ph_series.append(dict(phi=float(phi), W=W.tolist(), metrics=metrics(rho)))
channels_data['phase_sweep'] = ph_series

# Loss sweep
print("  Loss (Lindblad) sweep...")
loss_series = []
for gamma_t in [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]:
    rho = rho_base if gamma_t==0 else apply_loss(rho_base, gamma_t)
    W   = wigner_arr(rho)
    loss_series.append(dict(gamma_t=float(gamma_t), W=W.tolist(), metrics=metrics(rho)))
channels_data['loss_sweep'] = loss_series

channels_data['xvec'] = XVEC.tolist()
with open(OUT/"channels.pkl","wb") as f:
    pickle.dump(channels_data, f, protocol=4)
print(f"  ✅  data/channels.pkl  ({(OUT/'channels.pkl').stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
print("\n=== Generating GBS data ===")

gbs_data = {}

# Analytic GBS (no SF required — exact Gaussian state computation)
# For N squeezed modes through random interferometer
np.random.seed(42)

def generate_gbs_analytic(N_modes, r_vals):
    """Pure NumPy GBS: compute photon statistics analytically."""
    # Mean photon number per mode = sinh²(r)
    mn_per  = [float(np.sinh(r)**2) for r in r_vals]
    mq_vals = [float(2*np.sinh(r)**2 + 1) for r in r_vals]  # Mandel Q for squeezed
    
    # Marginal Wigner for each squeezed mode
    wigners = {}
    xvec_sf = np.linspace(-5, 5, 80)
    for i, r in enumerate(r_vals):
        # Squeezed vacuum Wigner
        sx = np.exp(-r); sp = np.exp(r)
        X, P = np.meshgrid(xvec_sf, xvec_sf)
        W = (2/np.pi) * np.exp(-2*(X**2*sx**2 + P**2*sp**2)) / (sx*sp)
        wigners[i] = W.tolist()
    
    # Photon number distribution (single mode, thermal-like after tracing)
    nbar  = np.sinh(r_vals[0])**2
    ns    = np.arange(25)
    if nbar > 0:
        p_n = (nbar/(1+nbar))**ns / (1+nbar)
    else:
        p_n = np.zeros(25); p_n[0]=1.0
    
    # Hafnian complexity curve
    n_arr  = np.arange(2, 22, 2)
    t_ryser = (2**n_arr * n_arr**2).tolist()
    
    return dict(
        mean_photons=mn_per, mq_vals=mq_vals,
        wigners=wigners, xvec=xvec_sf.tolist(),
        photon_dist=p_n.tolist(), photon_ns=ns.tolist(),
        hafnian_n=n_arr.tolist(), hafnian_ops=t_ryser,
        r_vals=[float(r) for r in r_vals], N_modes=N_modes,
    )

# Try real SF first
SF_OK = False
try:
    import strawberryfields as sf
    from strawberryfields import ops as sf_ops
    from strawberryfields.utils import random_interferometer
    
    print("  Strawberry Fields available — running real GBS...")
    for N_modes, r_uniform in [(4, 1.0), (6, 1.2)]:
        r_vals_sf = [r_uniform]*N_modes
        U  = random_interferometer(N_modes)
        prog = sf.Program(N_modes)
        with prog.context as q:
            for i, r in enumerate(r_vals_sf):
                sf_ops.Squeezed(r, 0) | q[i]
            sf_ops.Interferometer(U) | tuple(q[i] for i in range(N_modes))
        eng    = sf.Engine("gaussian")
        result = eng.run(prog)
        state  = result.state
        mn_per = [state.mean_photon(i)[0] for i in range(N_modes)]
        xvec_sf = np.linspace(-5, 5, 80)
        wigners = {}
        for i in range(N_modes):
            wigner_result = state.wigner(i, xvec_sf, xvec_sf)
            # state.wigner() returns (W, grid) tuple in some SF versions
            W_ = wigner_result[0] if isinstance(wigner_result, tuple) else wigner_result
            wigners[i] = W_.tolist()
        nbar   = mn_per[0]
        ns     = np.arange(25)
        if nbar > 0:
            p_n = (nbar / (1 + nbar)) ** ns / (1 + nbar)
        else:
            p_n = np.zeros(25)
            p_n[0] = 1.0
        n_arr  = np.arange(2, 22, 2)
        gbs_data[f"sf_{N_modes}"] = dict(
            mean_photons=[float(v) for v in mn_per], mq_vals=[float(2*v+1) for v in mn_per],
            wigners=wigners, xvec=xvec_sf.tolist(),
            photon_dist=p_n.tolist(), photon_ns=ns.tolist(),
            hafnian_n=n_arr.tolist(), hafnian_ops=(2**n_arr*n_arr**2).tolist(),
            r_vals=[r_uniform]*N_modes, N_modes=N_modes, source="strawberryfields",
        )
    SF_OK = True
    print("  ✅  Real GBS data saved")
except Exception as e:
    print(f"  ⚠️  SF not available ({e}) — using analytic GBS")

# Always generate analytic versions too (fast, no deps)
for N_modes, r_base in [(4, 1.0), (6, 1.2), (8, 0.8)]:
    r_vals = [r_base + 0.1*i for i in range(N_modes)]
    key = f"analytic_{N_modes}"
    gbs_data[key] = generate_gbs_analytic(N_modes, r_vals)
    gbs_data[key]['source'] = 'analytic'
    print(f"  Analytic GBS {N_modes} modes: done")

# Hafnian brute-force table
def hafnian_brute(A):
    n2 = A.shape[0]
    if n2 % 2 != 0: return 0.0
    idx = list(range(n2))
    haf = 0.0+0j
    def matchings(lst):
        if not lst: yield []; return
        first, rest = lst[0], lst[1:]
        for i, v in enumerate(rest):
            for m in matchings(rest[:i]+rest[i+1:]):
                yield [(first,v)]+m
    for m in matchings(idx):
        term=1.0+0j
        for (i,j) in m: term*=A[i,j]
        haf+=term
    return float(np.real(haf))

haf_table = []
for name_, A_ in [("2×2 Identity",np.eye(2)),("2×2 Ones",np.ones((2,2))),
                   ("4×4 Random",np.random.RandomState(42).randn(4,4))]:
    A_sym = (A_+A_.T)/2
    haf_table.append({"Matrix":name_,"Brute-force":round(hafnian_brute(A_sym),6)})
gbs_data['hafnian_table'] = haf_table

# CV-QML training curve
theta = np.linspace(-math.pi, math.pi, 200)
gbs_data['qml_theta']    = theta.tolist()
gbs_data['qml_exp_x']    = (1.5*np.sqrt(2)*np.cos(theta)).tolist()
gbs_data['qml_gradient'] = (-1.5*np.sqrt(2)*np.sin(theta)).tolist()
steps = np.arange(120)
loss  = np.maximum(2.5*np.exp(-steps/30)+0.05*np.random.RandomState(42).randn(120)*np.exp(-steps/60), 0.001)
gbs_data['qml_steps'] = steps.tolist()
gbs_data['qml_loss']  = loss.tolist()

with open(OUT/"gbs.pkl","wb") as f:
    pickle.dump(gbs_data, f, protocol=4)
print(f"  ✅  data/gbs.pkl  ({(OUT/'gbs.pkl').stat().st_size//1024} KB)")

print("\n✅ All data generated successfully!")
print("   data/states.pkl")
print("   data/channels.pkl")
print("   data/gbs.pkl")
print("\nNext: commit the data/ folder + app_static.py to GitHub, deploy on Streamlit Cloud.")
