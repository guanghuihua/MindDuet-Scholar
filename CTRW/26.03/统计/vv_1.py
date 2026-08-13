"""
canard_escape_comparison.py
============================
Compare Euler-Maruyama (EM) vs CTMC/SSA numerical schemes for a canard system:
statistics of trajectories crawling along the unstable slow manifold and
reaching a target point x*.

System equations (FitzHugh-Nagumo / van der Pol canard family):
    dx = (y - x^3/3 + x) / delta * dt + eps * dW_x
    dy = (a - x)           * dt + eps * dW_y
    delta = 0.1,  a = a_canard (maximal canard parameter)

Experiment logic:
    Each trial starts on the right stable branch; once the trajectory
    enters the unstable branch region, we record whether it reaches x*
    and where it eventually escapes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from tqdm import tqdm

# ── System parameters ────────────────────────────────────────────────────────
DELTA = 0.1
A = 1 - DELTA/8 - 3*DELTA**2/32 - 173*DELTA**3/1024 - 0.01  # canard value

# ── Experiment hyperparameters ───────────────────────────────────────────────
EPS        = 0.08    # noise intensity
N_TRIALS   = 200     # number of trials per scheme
X_TARGET   = 0.0     # target point on unstable branch, x in (-1, 1)
TARGET_TOL = 0.12    # arrival criterion: |x - x*| < TARGET_TOL

# EM parameters
DT = 0.003

# SSA/CTMC parameters
H = 0.06             # lattice spacing

# Detection parameters
UNSTABLE_TOL  = 0.25   # distance to nullcline for "on unstable branch"
UNSTABLE_XMIN = -1.05
UNSTABLE_XMAX =  1.05

# ── Helper functions ─────────────────────────────────────────────────────────
def nullcline_y(x):
    """x-nullcline: y = x^3/3 - x"""
    return x**3 / 3 - x


def randn2():
    return np.random.randn(), np.random.randn()


# ── EM single trial ──────────────────────────────────────────────────────────
def run_em_trial(eps=EPS, dt=DT, x_target=X_TARGET, max_steps=500_000):
    """
    Returns (reached: bool, exit_x: float, trajectory: list[(x, y)]).
    Trajectory is recorded only while on the unstable branch to save memory.
    """
    x = 1.8
    y = nullcline_y(1.8)
    sqdt = np.sqrt(dt)

    reached     = False
    exit_x      = None
    on_unstable = False
    traj        = []

    for _ in range(max_steps):
        mu1 = (y - x**3/3 + x) / DELTA
        mu2 = A - x
        xi1, xi2 = randn2()
        x += mu1*dt + eps*sqdt*xi1
        y += mu2*dt + eps*sqdt*xi2

        if abs(x) > 2.8 or abs(y) > 2.8:
            break

        dist        = abs(y - nullcline_y(x))
        in_unstable = UNSTABLE_XMIN < x < UNSTABLE_XMAX

        if in_unstable and dist < UNSTABLE_TOL:
            on_unstable = True
            traj.append((x, y))
            if abs(x - x_target) < TARGET_TOL:
                reached = True
        elif on_unstable:
            exit_x = np.clip(x, -1.0, 1.0) if not in_unstable else x
            break

    if exit_x is None:
        exit_x = x
    return reached, exit_x, traj


# ── SSA/CTMC single trial ─────────────────────────────────────────────────────
def run_ssa_trial(eps=EPS, h=H, x_target=X_TARGET, max_steps=300_000):
    """
    CTMC/SSA (Gillespie) scheme.
    Jump rates = upwind drift + local-averaging diffusion.
    Returns (reached, exit_x, trajectory).
    """
    x = round(1.8 / h) * h
    y = round(nullcline_y(1.8) / h) * h

    reached     = False
    exit_x      = None
    on_unstable = False
    traj        = []

    for _ in range(max_steps):
        mu1 = (y - x**3/3 + x) / DELTA
        mu2 = A - x

        m1 = max(2 - abs(mu1)*h, 0.0) / 2.0
        m2 = max(2 - abs(mu2)*h, 0.0) / 2.0

        q0 = max( mu1, 0) / h + m1 / h**2   # x -> x+h
        q1 = max(-mu1, 0) / h + m1 / h**2   # x -> x-h
        q2 = max( mu2, 0) / h + m2 / h**2   # y -> y+h
        q3 = max(-mu2, 0) / h + m2 / h**2   # y -> y-h
        lam = q0 + q1 + q2 + q3

        if lam <= 0:
            break

        r1 = np.random.random()
        # tau not needed here; only the jump direction matters
        if   r1*lam < q0:          x += h
        elif r1*lam < q0+q1:       x -= h
        elif r1*lam < q0+q1+q2:    y += h
        else:                       y -= h

        if abs(x) > 2.8 or abs(y) > 2.8:
            break

        dist        = abs(y - nullcline_y(x))
        in_unstable = UNSTABLE_XMIN < x < UNSTABLE_XMAX

        if in_unstable and dist < UNSTABLE_TOL:
            on_unstable = True
            traj.append((x, y))
            if abs(x - x_target) < TARGET_TOL:
                reached = True
        elif on_unstable:
            exit_x = np.clip(x, -1.0, 1.0) if not in_unstable else x
            break

    if exit_x is None:
        exit_x = x
    return reached, exit_x, traj


# ── Batch runner ──────────────────────────────────────────────────────────────
def run_batch(method, n_trials):
    hits         = 0
    exits        = []
    sample_trajs = []   # keep a few trajectories for visualization
    fn = run_em_trial if method == 'em' else run_ssa_trial

    for _ in tqdm(range(n_trials), desc=f'{method.upper():4s}', ncols=60):
        reached, exit_x, traj = fn()
        if reached:
            hits += 1
        exits.append(exit_x)
        if len(sample_trajs) < 8 and len(traj) > 3:
            sample_trajs.append((reached, traj))

    return hits, np.array(exits), sample_trajs


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_comparison(em_hits,  em_exits,  em_trajs,
                    ssa_hits, ssa_exits, ssa_trajs):

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor('#fafaf8')
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.07, right=0.97,
                           top=0.91,  bottom=0.09,
                           wspace=0.35, hspace=0.45)

    ax_phase = fig.add_subplot(gs[:, 0])   # phase portrait (full column)
    ax_em    = fig.add_subplot(gs[0, 1])   # EM sample trajectories
    ax_ssa   = fig.add_subplot(gs[0, 2])   # SSA sample trajectories
    ax_hist  = fig.add_subplot(gs[1, 1])   # escape-point histogram
    ax_stats = fig.add_subplot(gs[1, 2])   # summary bar chart

    col_em             = '#534AB7'
    col_ssa            = '#1D9E75'
    col_nullc_stable   = '#BA7517'
    col_nullc_unstable = '#A32D2D'
    col_ynullc         = '#0F6E56'
    bg                 = '#fafaf8'

    for ax in [ax_phase, ax_em, ax_ssa, ax_hist, ax_stats]:
        ax.set_facecolor(bg)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color('#bbbbbb')

    x_arr = np.linspace(-2.1, 2.1, 800)
    y_nc  = nullcline_y(x_arr)

    def draw_nullclines(ax, alpha=1.0):
        mask_l = x_arr <= -1
        mask_r = x_arr >=  1
        mask_u = (x_arr >= -1) & (x_arr <= 1)
        # stable branches (left and right)
        ax.plot(x_arr[mask_l], y_nc[mask_l],
                color=col_nullc_stable, lw=1.8, alpha=alpha, zorder=3)
        ax.plot(x_arr[mask_r], y_nc[mask_r],
                color=col_nullc_stable, lw=1.8, alpha=alpha, zorder=3,
                label='Stable slow manifold')
        # unstable branch (dashed)
        ax.plot(x_arr[mask_u], y_nc[mask_u],
                color=col_nullc_unstable, lw=2, ls='--', alpha=alpha,
                zorder=3, label='Unstable branch')
        # y-nullcline
        ax.axvline(A, color=col_ynullc, lw=1.2, ls=':', alpha=0.7*alpha,
                   label=f'y-nullcline (x=a)')
        # saddle points
        for sx in [-1, 1]:
            ax.plot(sx, nullcline_y(sx), 'o',
                    color=col_nullc_stable, ms=5, zorder=5)

    # ── Phase portrait ───────────────────────────────────────────────────────
    draw_nullclines(ax_phase)

    ax_phase.scatter(em_exits,  nullcline_y(np.clip(em_exits,  -1.05, 1.05)),
                     c=col_em,  alpha=0.45, s=18, zorder=6, label='EM escape points')
    ax_phase.scatter(ssa_exits, nullcline_y(np.clip(ssa_exits, -1.05, 1.05)),
                     c=col_ssa, alpha=0.45, s=18, zorder=6,
                     label='SSA escape points', marker='^')

    ty   = nullcline_y(X_TARGET)
    circ = Circle((X_TARGET, ty), 0.12, fc='none', ec='#E24B4A', lw=2, zorder=7)
    ax_phase.add_patch(circ)
    ax_phase.annotate(f'target x*={X_TARGET:.2f}',
                      xy=(X_TARGET, ty), xytext=(X_TARGET+0.35, ty+0.3),
                      fontsize=9, color='#A32D2D',
                      arrowprops=dict(arrowstyle='->', color='#A32D2D', lw=1))

    ax_phase.set_xlim(-2.1, 2.1)
    ax_phase.set_ylim(-1.9, 1.9)
    ax_phase.set_xlabel('x', fontsize=11)
    ax_phase.set_ylabel('y', fontsize=11)
    ax_phase.set_title('Phase portrait: escape point distribution',
                       fontsize=12, fontweight='normal')
    ax_phase.legend(fontsize=8, loc='upper left', framealpha=0.85)
    ax_phase.tick_params(labelsize=9)

    # ── Sample trajectories ──────────────────────────────────────────────────
    def draw_sample_trajs(ax, trajs, color, title):
        draw_nullclines(ax, alpha=0.6)
        for reached, traj in trajs:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            lw = 1.2 if reached else 0.7
            al = 0.75 if reached else 0.35
            ax.plot(xs, ys, color=color, lw=lw, alpha=al)
            ax.plot(xs[-1], ys[-1], 'x', color=color, ms=5, alpha=0.8)
        ty_t = nullcline_y(X_TARGET)
        ax.plot(X_TARGET, ty_t, 'o', color='#E24B4A', ms=7, zorder=8,
                label=f'x*={X_TARGET:.2f}')
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.4, 1.0)
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='normal')
        ax.legend(fontsize=8, loc='upper left', framealpha=0.85)
        ax.tick_params(labelsize=9)

    draw_sample_trajs(ax_em,  em_trajs,  col_em,
                      f'EM sample trajectories (dt={DT})')
    draw_sample_trajs(ax_ssa, ssa_trajs, col_ssa,
                      f'SSA sample trajectories (h={H})')

    # ── Escape-point histogram ───────────────────────────────────────────────
    bins = np.linspace(-1.1, 1.1, 24)
    ax_hist.hist(em_exits,  bins=bins, color=col_em,  alpha=0.65,
                 label='EM',  density=True)
    ax_hist.hist(ssa_exits, bins=bins, color=col_ssa, alpha=0.65,
                 label='SSA', density=True)
    ax_hist.axvline(X_TARGET, color='#E24B4A', lw=1.5, ls='--',
                    label=f'x*={X_TARGET:.2f}')
    ax_hist.axvline(np.mean(em_exits),  color=col_em,  lw=1.2, ls=':')
    ax_hist.axvline(np.mean(ssa_exits), color=col_ssa, lw=1.2, ls=':')
    ax_hist.set_xlabel('Escape point x coordinate', fontsize=10)
    ax_hist.set_ylabel('Density', fontsize=10)
    ax_hist.set_title('Escape point distribution', fontsize=11, fontweight='normal')
    ax_hist.legend(fontsize=9, framealpha=0.85)
    ax_hist.tick_params(labelsize=9)

    # ── Summary bar chart ────────────────────────────────────────────────────
    em_prob  = em_hits  / N_TRIALS * 100
    ssa_prob = ssa_hits / N_TRIALS * 100

    methods = ['EM', 'SSA']
    probs   = [em_prob, ssa_prob]
    colors  = [col_em, col_ssa]
    bars = ax_stats.bar(methods, probs, color=colors, width=0.45,
                        alpha=0.85, edgecolor='none')
    for bar, p, h_v in zip(bars, probs, [em_hits, ssa_hits]):
        ax_stats.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 1.0,
                      f'{p:.1f}%\n({h_v}/{N_TRIALS})',
                      ha='center', va='bottom', fontsize=11, fontweight='500')

    em_mean  = np.mean(em_exits)
    ssa_mean = np.mean(ssa_exits)
    info = (f'EM  mean escape x={em_mean:.3f}\n'
            f'SSA mean escape x={ssa_mean:.3f}')
    ax_stats.text(0.5, -0.22, info, transform=ax_stats.transAxes,
                  ha='center', fontsize=9, color='#555',
                  bbox=dict(fc='#f0f0ee', ec='none', pad=4))

    ax_stats.set_ylim(0, max(probs)*1.35 + 5)
    ax_stats.set_ylabel('Probability of reaching x* (%)', fontsize=10)
    ax_stats.set_title('Arrival probability at target', fontsize=11,
                       fontweight='normal')
    ax_stats.tick_params(labelsize=10)

    # ── Overall title ────────────────────────────────────────────────────────
    fig.suptitle(
        f'Canard system: EM vs SSA/CTMC numerical scheme comparison\n'
        f'eps={EPS},  x*={X_TARGET},  N={N_TRIALS} trials,  '
        f'delta={DELTA},  a={A:.4f}',
        fontsize=13, fontweight='normal', y=0.975
    )

    plt.savefig('canard_escape_comparison.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('canard_escape_comparison.png', dpi=150, bbox_inches='tight')
    print('Figures saved: canard_escape_comparison.pdf / .png')
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(42)
    print(f'System parameters: delta={DELTA}, a={A:.6f}')
    print(f'Experiment parameters: eps={EPS}, N={N_TRIALS}, x*={X_TARGET}')
    print(f'EM:  dt={DT}')
    print(f'SSA: h={H}')
    print()

    em_hits,  em_exits,  em_trajs  = run_batch('em',  N_TRIALS)
    ssa_hits, ssa_exits, ssa_trajs = run_batch('ssa', N_TRIALS)

    print(f'\nResults:')
    print(f'  EM  reached x*={X_TARGET}: {em_hits}/{N_TRIALS} = {em_hits/N_TRIALS*100:.1f}%')
    print(f'  SSA reached x*={X_TARGET}: {ssa_hits}/{N_TRIALS} = {ssa_hits/N_TRIALS*100:.1f}%')
    print(f'  EM  mean escape position: x = {np.mean(em_exits):.4f}')
    print(f'  SSA mean escape position: x = {np.mean(ssa_exits):.4f}')

    plot_comparison(em_hits,  em_exits,  em_trajs,
                    ssa_hits, ssa_exits, ssa_trajs)