import meshio
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#Konstanter
a0    = 0.01   # m
k     = 1e2    # N/m  (= κ·B, med B=1 m)
K     = 1e4    # N/m
B     = 1.0    # m (dybde inn i planet)
kappa = k / B

E_analytisk = 8 * kappa / np.sqrt(3)
v_analytisk = 1 / 3


#Lesing av mesh fra fil
def read_mesh(lx0, ly0, n_target):
    mesh  = meshio.read(f"meshes/mesh_Lx{lx0}_Ly{ly0}_Ntarget{n_target}.vtk")
    xy    = mesh.points[:, :2]
    edges = mesh.cells_dict["line"]
    return xy, edges


#Oppgave 2: Bygg enkelt nettverk 
def make_simple_mesh(a0):
    b0 = np.sqrt(3) * a0
    xy = np.array([
        [0,    0   ],
        [b0,   0   ],
        [b0/2, a0/2],
        [0,    a0  ],
        [b0,   a0  ],
    ])
    edges = np.array([
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 4],
        [2, 3],
        [2, 4],
        [3, 4],
    ])
    return xy, edges


def plot_mesh(xy, edges):
    fig, ax = plt.subplots(1, 1)
    for edge in edges:
        ax.plot(xy[edge, 0], xy[edge, 1], 'k')
    ax.scatter(xy[:, 0], xy[:, 1], s=20, color='red')
    ax.set_aspect("equal")
    plt.show()


xy0, edges = make_simple_mesh(a0)
plot_mesh(xy0, edges)

#Beregner hvilelengden til hver fjær
ell0 = np.linalg.norm(xy0[edges[:, 0]] - xy0[edges[:, 1]], axis=1)
print(ell0)


#Oppgave 3: Energi og krefter
def spring_energy(xy, edges, k, ell0):
    energy = 0.0
    for edge, l0 in zip(edges, ell0):
        i, j  = edge
        rij   = xy[j] - xy[i]
        ell   = np.linalg.norm(rij)
        energy += 0.5 * k * (ell - l0) ** 2
    return energy

# Setter inn mesh i hviletilstand – forventer null energi
print(spring_energy(xy0, edges, k, ell0))


def spring_forces(xy, edges, k, ell0):
    forces = np.zeros_like(xy)
    for edge, l0 in zip(edges, ell0):
        i, j = edge
        rij  = xy[j] - xy[i]
        ell  = np.linalg.norm(rij)
        if ell > 0:
            f_mag = k * (ell - l0)
            f_vec = f_mag * (rij / ell)
            forces[i, :] += f_vec
            forces[j, :] -= f_vec
    return forces


# Oppgave 4: Randindekser og total energi
lx0 = np.sqrt(3) * a0
ly0 = a0

ids_left   = xy0[:, 0] < 1e-12
ids_right  = xy0[:, 0] > lx0 - 1e-12
ids_bottom = xy0[:, 1] < 1e-12
ids_top    = xy0[:, 1] > ly0 - 1e-12


def total_energy(xy_flat, edges, k, K, ell0, lx_plate):
    xy     = xy_flat.reshape((-1, 2))
    energy = spring_energy(xy, edges, k, ell0)
    # Potensiell energi lagret i klamme på venstre side
    energy += 0.5 * K * ((xy[ids_left,  0]) ** 2).sum()
    # Potensiell energi lagret i klamme på høyre side
    energy += 0.5 * K * ((xy[ids_right, 0] - lx_plate) ** 2).sum()
    # Klamme som hindrer massesenteret fra å flytte seg i y-retning
    energy += 0.5 * K * ((xy[ids_left,  1] - xy0[ids_left,  1]).mean() ** 2)
    energy += 0.5 * K * ((xy[ids_right, 1] - xy0[ids_right, 1]).mean() ** 2)
    return energy


# Oppgave 5:
def total_energy_jacobian(xy_flat, edges, k, K, ell0, lx_plate):
    xy   = xy_flat.reshape((-1, 2))
    grad = -spring_forces(xy, edges, k, ell0)
    grad[ids_left,  0] += K * xy[ids_left, 0]
    grad[ids_right, 0] += K * (xy[ids_right, 0] - lx_plate)
    grad[ids_left,  1] += K * (xy[ids_left,  1] - xy0[ids_left,  1]).mean()
    grad[ids_right, 1] += K * (xy[ids_right, 1] - xy0[ids_right, 1]).mean()
    return grad.flatten()


def plot_equilibrium(xy, edges, k, ell0, title="Likevekt"):
    fig, ax = plt.subplots(figsize=(8, 4))

    p1              = xy[edges[:, 0]]
    p2              = xy[edges[:, 1]]
    current_lengths = np.linalg.norm(p1 - p2, axis=1)

    # Positiv verdi = strekk (rød), negativ = kompresjon (blå)
    forces = k * (current_lengths - ell0)
    max_f  = np.max(np.abs(forces)) if np.max(np.abs(forces)) > 1e-10 else 1.0

    for i, edge in enumerate(edges):
        color = plt.cm.coolwarm(0.5 + 0.5 * forces[i] / max_f)
        ax.plot(xy[edge, 0], xy[edge, 1], color=color, lw=2)

    ax.scatter(xy[:, 0], xy[:, 1], s=30, color='black', zorder=3)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.coolwarm,
        norm=plt.Normalize(-max_f, max_f),
    )
    plt.colorbar(sm, ax=ax, label="Kraft [N]  (blå: kompresjon, rød: strekk)")
    plt.show()


#like strekkfaktorer - 10%, 20%, 30%
f_values = [0.1, 0.2, 0.3]

print(f"{'f':>8} | {'Lx':>10} | {'Ly':>10} | {'E_sim':>10} | {'v_sim':>10}")
print("-" * 55)

for f in f_values:
    lx_plate = lx0 * (1 + f)

    res = minimize(
        total_energy,
        xy0.flatten(),
        args=(edges, k, K, ell0, lx_plate),
        method='Newton-CG',
        jac=total_energy_jacobian,
        tol=1e-12,
    )

    xy = res.x.reshape((-1, 2))
    lx = xy[ids_right, 0].mean() - xy[ids_left,  0].mean()
    ly = xy[ids_top,   1].mean() - xy[ids_bottom, 1].mean()

    fn    = -K * (xy[ids_right, 0] - lx_plate).sum()
    eps_n = (lx - lx0) / lx0
    eps_t = (ly - ly0) / ly0
    sigma = fn / (ly0 * B)

    e_sim = sigma / eps_n
    v_sim = -eps_t / eps_n

    print(f"{f:8.3f} | {lx:10.4f} | {ly:10.4f} | {e_sim:10.4f} | {v_sim:10.4f}")

    plot_equilibrium(xy, edges, k, ell0, title=f"Oppgave 5: Likevekt ved f={f}")



# Oppgave 6: Mål E og ν som funksjon av εn

f_values = np.linspace(0.001, 1, 40) #40 punkter mellom 0.001 og 1

eps_n_list = [] #epsilon_n
E_sim_list = [] #Youngs modul
v_sim_list = [] #Poissons tall

for f in f_values:
    lx_plate = lx0 * (1 + f)

    res = minimize(
        total_energy,
        xy0.flatten(),
        args=(edges, k, K, ell0, lx_plate),     #Finner energiminimumet og lagrer koordiatene i res.x inni res-"biblioteket"
        method='Newton-CG',
        jac=total_energy_jacobian,
        tol=1e-12,
        options={'maxiter': 1000},
    )

    xy_eq = res.x.reshape((-1, 2)) #En N*2 koordinat-array når systemet er i likevekt
    lx    = xy_eq[ids_right, 0].mean() - xy_eq[ids_left,   0].mean()
    ly    = xy_eq[ids_top,   1].mean() - xy_eq[ids_bottom,  1].mean()

    fn    = -K * (xy_eq[ids_right, 0] - lx_plate).sum()
    eps_n = (lx - lx0) / lx0
    eps_t = (ly - ly0) / ly0
    sigma = fn / (ly0 * B)

    eps_n_list.append(eps_n)
    E_sim_list.append(sigma / eps_n)
    v_sim_list.append(-eps_t / eps_n)

eps_n_arr = np.array(eps_n_list)
E_sim_arr = np.array(E_sim_list)
v_sim_arr = np.array(v_sim_list)

# Plot Young som funksjon av epsilon_n
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(eps_n_arr, E_sim_arr, 'o-', label='Simulert $E$')
ax1.axhline(E_analytisk, color='red', linestyle='--',
            label=f'Analytisk $E = 8\\kappa/\\sqrt{{3}}$ = {E_analytisk:.2f}')
ax1.set_xlabel('Relativt strekk $\\varepsilon_n$')
ax1.set_ylabel('Youngs modul E')
ax1.set_title("E($\\varepsilon_n$)")
ax1.legend()
ax1.grid(True)

# Plot Poisson som funksjon av epsilon_n
ax2.plot(eps_n_arr, v_sim_arr, 'o-', label="Simulert $\\nu$")
ax2.axhline(v_analytisk, color='red', linestyle='--',
            label=f'Analytisk $\\nu = 1/3$ = {v_analytisk:.4f}')
ax2.set_xlabel('Relativt strekk $\\varepsilon_n$')
ax2.set_ylabel("Poissons tall $\\nu$")
ax2.set_title("$\\nu$($\\varepsilon_n$)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

#oppgave 7

def plot_mesh(xy, edges, title="Mesh"):
    fig, ax = plt.subplots()
    for edge in edges:
        ax.plot(xy[edge, 0], xy[edge, 1], 'k-', lw=0.5)
    ax.scatter(xy[:, 0], xy[:, 1], s=10, color='red')
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.show()

xy0, Edges = read_mesh(lx0, ly0, N)
ell0_      = np.linalg.norm(xy0[Edges[:, 0]] - xy0[Edges[:, 1]], axis=1)
plot_mesh(xy0, Edges, title="Udeformert mesh")

ids_left   = xy0[:, 0] < 1e-10
ids_right  = xy0[:, 0] > lx0 - 1e-10
ids_bottom = xy0[:, 1] < 1e-10
ids_top    = xy0[:, 1] > ly0 - 1e-10

# --- Fjærenergier (vektorisert) ---
def spring_energy(xy, edges, k, ell0_):
    dr  = xy[edges[:, 1]] - xy[edges[:, 0]]
    ell = np.linalg.norm(dr, axis=1)
    return 0.5 * k * np.sum((ell - ell0_)**2)

def spring_forces(xy, edges, k, ell0_):
    forces = np.zeros_like(xy)
    dr     = xy[edges[:, 1]] - xy[edges[:, 0]]
    ell    = np.linalg.norm(dr, axis=1, keepdims=True)
    f_vec  = k * (ell - ell0_[:, None]) * (dr / ell)
    np.add.at(forces, edges[:, 0],  f_vec)
    np.add.at(forces, edges[:, 1], -f_vec)
    return forces

def spring_strains(xy, edges, ell0_):
    dr  = xy[edges[:, 1]] - xy[edges[:, 0]]
    ell = np.linalg.norm(dr, axis=1)
    return (ell - ell0_) / ell0_

# --- Total energi og jakobisk ---
def total_energy(xy_flat, edges, k, K, ell0_, Lx_plate):
    xy     = xy_flat.reshape((-1, 2))
    energy = spring_energy(xy, edges, k, ell0_)
    energy += 0.5 * K * (xy[ids_left,  0]**2).sum()
    energy += 0.5 * K * ((xy[ids_right, 0] - Lx_plate)**2).sum()
    energy += 0.5 * K * (xy[ids_left,  1] - xy0[ids_left,  1]).mean()**2
    energy += 0.5 * K * (xy[ids_right, 1] - xy0[ids_right, 1]).mean()**2
    return energy

def total_energy_jacobian(xy_flat, edges, k, K, ell0_, Lx_plate):
    xy   = xy_flat.reshape((-1, 2))
    grad = -spring_forces(xy, edges, k, ell0_)
    grad[ids_left,  0] += K * xy[ids_left, 0]
    grad[ids_right, 0] += K * (xy[ids_right, 0] - Lx_plate)
    grad[ids_left,  1] += K * (xy[ids_left,  1] - XY0[ids_left,  1]).mean()
    grad[ids_right, 1] += K * (xy[ids_right, 1] - XY0[ids_right, 1]).mean()
    return grad.flatten()

MINIMIZE_KWARGS = dict(
    method='Newton-CG',
    jac=total_energy_jacobian,
    tol=1e-12,
    options={'maxiter': 1000}
)

# --- Visualisering av deformert mesh med strekk/kompresjon ---
def plot_deformed(xy, edges, ell0_, title="Deformert mesh"):
    strains = spring_strains(xy, edges, ell0_)
    vmax    = np.max(np.abs(strains))

    fig, ax = plt.subplots(figsize=(10, 5))
    sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                                norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    for edge, strain in zip(edges, strains):
        color = sm.cmap(sm.norm(strain))
        ax.plot(xy[edge, 0], xy[edge, 1], color=color, lw=1.5)

    ax.scatter(xy[:, 0], xy[:, 1], s=10, color='k', zorder=5)
    plt.colorbar(sm, ax=ax, label='Tøyning (strekk > 0, kompresjon < 0)')
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# --- Plott ved ulike tøyninger ---
f_plot = [0.01, 0.05, 0.1, 0.3]
xy = xy0.copy()

for f in f_plot:
    lx_target = Lx0 * (1 + f)
    res = minimize(total_energy, xy.flatten(),
                   args=(Edges, k, K, ell0_, lx_target), **MINIMIZE_KWARGS)
    xy = res.x.reshape((-1, 2))
    plot_deformed(xy, Edges, ell0_, title=f"Deformert mesh, f={f:.2f}  (L={lx_target:.4f} m)")

# --- E og ν over spekter av tøyninger ---
f_values = np.logspace(-3, 0, 30)
res_list = []
xy = xy0.copy()

for f in f_values:
    lx_target = lx0 * (1 + f)
    res = minimize(total_energy, XY.flatten(),
                   args=(Edges, k, K, ell0_, lx_target), **MINIMIZE_KWARGS)
    XY = res.x.reshape((-1, 2))

    lx = xy[ids_right, 0].mean() - xy[ids_left,   0].mean()
    ly = xy[ids_top,   1].mean() - xy[ids_bottom,  1].mean()
    en = (lx - lx0) / lx0
    et = (ly - ly0) / ly0
    sn = -K * (XY[ids_right, 0] - lx_target).sum() / (ly0 * B)
    res_list.append([en, sn / en, -et / en])

en_arr, E_arr, nu_arr = np.array(res_list).T

# --- Plotting E og ν ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, data, theory, title, ylabel in zip(
    axes,
    [E_arr,    nu_arr],
    [E_theory, nu_theory],
    [f'Youngs modul (Teori: {E_theory:.2f})', f'Poissons tall (Teori: {nu_theory:.3f})'],
    ['E [N/m²]', 'ν']
):
    ax.semilogx(en_arr, data, 'o-', label='Simulert')
    ax.axhline(theory, color='r', ls='--', label=f'Teori ({theory:.3f})')
    ax.set(xlabel='εn', ylabel=ylabel, title=title)
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

print(f"Resultater ved minste εn:")
print(f"  E = {E_arr[0]:.4f}  (teori: {E_theory:.4f})")
print(f"  ν = {nu_arr[0]:.4f}  (teori: {nu_theory:.4f})")