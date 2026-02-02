import tkinter as tk
from tkinter import ttk

import numpy as np
import scipy.linalg as la

import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import sys

#-------------Spline Definitions-----------------------------

def cubic_bspline(u):
    '''  
    For each input u, computes the cubic B-spline basis function value.
    '''
    absu = jnp.abs(u)
    return jnp.where(
        absu < 1,
        (4 - 6*absu**2 + 3*absu**3) / 6,
        jnp.where(
            absu < 2,
            (2 - absu)**3 / 6,
            0.0
        )
    )

def make_spline(weights, L=1.0):
    '''  
    Makes a spline with the number of knots/nodes determined by how many weights are given.
    weights: array of shape (n,)
    L: length of the interval [0, L]
    
    Sums the piecewise basis functions defined in cubic_bspline, scaled by the weights.
    '''
    weights = jnp.asarray(weights)
    n = weights.shape[0]
    dx = L / (n - 1)
    x_nodes = jnp.linspace(0, L, n)

    def spline(x):
        u = (x[..., None] - x_nodes) / dx
        return jnp.sum(weights * cubic_bspline(u), axis=-1)

    return jax.jit(spline)



#-----------Initial Spline/Plotting-----------------------

weights = np.random.rand(10)

f = make_spline(weights)

x = np.linspace(0, 1, 501) # For the time being, L must be 1 and Nx=500

# Initial plot

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_ylim(-0.1, 1.5)
D_line, = ax.plot(x, f(x), linestyle='--', color='grey', label='Diffusion Coefficient D(x)')


#----------Tkinter Interface--------------------

def on_close():
    root.quit()
    root.destroy()
    sys.exit()

root = tk.Tk()
root.protocol("WM_DELETE_WINDOW", on_close)

frame = ttk.Frame(root, padding="10")
frame.pack(fill="both", expand=True)
frame.columnconfigure(1, weight=1)
frame.rowconfigure(0, weight=1)

# --------Moving Spline with Update Sliders------------

# Spline Plot

canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().grid(column=1, row=0)

def update_spline(slider_num, slider_value):
    weights[slider_num] = float(slider_value)
    f = make_spline(weights)
    D_line.set_ydata(f(x))
    canvas.draw_idle()

    # Update the label
    _, label = sliders[slider_num]
    label.config(text=f"{float(slider_value):.2f}")


#---------------Interface: Control panel-----------------


control_panel = ttk.LabelFrame(frame, text="Inputs", padding="10")
control_panel.grid(column=0, row=0)

# Sliders

sliders = {}
for slider_num in range(len(weights)):
    # Create label
    slider_label = ttk.Label(control_panel, text=f"Weight {slider_num}: ")
    slider_label.grid(column=0, row=slider_num+1)

    # Create Slider
    slider = ttk.Scale(
        control_panel, 
        from_=0, to=1, 
        orient='horizontal', 
        command=lambda v, n=slider_num: update_spline(n, v)
    )
    slider.set(weights[slider_num])
    slider.grid(column=1, row=slider_num+1)

    # Show value
    value_label = ttk.Label(control_panel, text=f"{weights[slider_num]:.2f}")
    value_label.grid(column=2, row=slider_num+1, padx=5)

    # Store both slider and value label for future updates
    sliders[slider_num] = (slider, value_label)

# Solver Inputs

input_box = ttk.LabelFrame(control_panel, text="Initial Conditions", padding="10")
input_box.grid(column=0, row=len(weights)+1, columnspan=3, pady=10)

# C0 [ng/cm^3]
C0_value = tk.DoubleVar(value=0)
# Left label (variable name)
C0_label = ttk.Label(input_box, text="C0: ")
C0_label.grid(row=0, column=0, padx=5)
# Entry (float)
C0_entrybox = ttk.Entry(input_box, textvariable=C0_value, width=10, justify="right")
C0_entrybox.grid(row=0, column=1, padx=5)
# Right label (units)
C0_unit = ttk.Label(input_box, text="ng/cm³")
C0_unit.grid(row=0, column=2, padx=5)

# L [A]
L_value = tk.DoubleVar(value=0)
# Left label (variable name)
L_label = ttk.Label(input_box, text="L: ")
L_label.grid(row=1, column=0, padx=5)
# Entry (float)
L_entrybox = ttk.Entry(input_box, textvariable=L_value, width=10, justify="right")
L_entrybox.grid(row=1, column=1, padx=5)
# Right label (units)
L_unit = ttk.Label(input_box, text="Å")
L_unit.grid(row=1, column=2, padx=5)

# tf [s]
tf_value = tk.DoubleVar(value=0)
# Left label (variable name)
tf_label = ttk.Label(input_box, text="tf: ")
tf_label.grid(row=2, column=0, padx=5)
# Entry (float)
tf_entrybox = ttk.Entry(input_box, textvariable=tf_value, width=10, justify="right")
tf_entrybox.grid(row=2, column=1, padx=5)
# Right label (units)
tf_unit = ttk.Label(input_box, text="s")
tf_unit.grid(row=2, column=2, padx=5)
# -----------------------------------

#-------------Solvers--------------------

def num_grid(C0, D, k, L, tf, Nx=500, Nt=500):
    '''
    Uses Crank-Nicolson propagation to find concentration profile over time. 
    Allows for variable diffusion coeficient D(x).
    Includes explicit reaction term k.
    Uses earlier defined experimental boundary conditions.

    C0: Constant Dirichlet concentration at top boundary.
    D: diffusion coefficient D(x)
    k: reaction rate constant
    L: length of rod
    tf: final time
    Nx: Number of space steps. Including the 0th point, returns a matrix with dimention Nx+1.
    Nt: Number of time steps. Including the 0th point, returns a matrix with dimention Nt+1.
    x_scale: Optional rescale of x grid points.
    '''

    dt = tf/Nt # t_0 = 0, t_1 = dt, t_2 = 2*dt, ...,t_n = n*dt, t_Nt = Nt*dt = tf
    dx = L/Nx # x_0 = 0, x_1 = dx, x_2 = 2*dx, ..., x_i = i*dx, x_Nx = Nx*dx = L

    xgrid = np.linspace(0, L, Nx+1)
    Di = D(xgrid) # Discretizes D(x) to a vector of Nx+1 elements

    # Initialize diffusion ratios
    # r_alpha, r_beta differ by 1 index/1 element at each end.
    # r_ab[0] = 1/2 * (D0+D1)*dt / 2*dx^2 = r_alpha[0] = r_beta[1]
    r_ab = 0.5*(Di[:-1] + Di[1:])*dt / (2*dx**2) # Size (Nx,) 
    
    #if k != 0 and dt > 2/k:
    #    tqdm.write(f"dt={dt:.5f} outside of stability conditions! (Nt={Nt}, k={k})")     

    A_bands = np.zeros((3,Nx+1))
    A_bands[0,1:] = -r_ab # Upper Diagonal: -r_alpha[0], -r_alpha[1], -r_alpha[2], ..., -r_alpha[Nx]
    A_bands[1,1:-1] = (1+r_ab[1:]+r_ab[:-1]) # Main Diagonal[i]: 1 + r_alpha[i] + r_alpha[i-1] = 1 + r_alpha[i] + r_beta[i]
    A_bands[1,0] = 1+2*r_ab[0] # Main diagonal endpoints (double r for conservation via ghost points)
    A_bands[1,-1] = 1+2*r_ab[-1]
    A_bands[2,:-1] = -r_ab # Lower Diagonal

    B_bands = np.zeros((3,Nx+1))
    B_bands[0,1:] = r_ab
    B_bands[1,1:-1] = (1-r_ab[1:]-r_ab[:-1])
    B_bands[1,0] = 1-2*r_ab[0]
    B_bands[1,-1] = 1-2*r_ab[-1]
    B_bands[2,:-1] = r_ab

    B = np.diag(B_bands[1,:]) + np.diag(B_bands[0,1:], k=+1) + np.diag(B_bands[2,:-1], k=-1)

    # Lext/top Dirichlet boundary condition:
    # Set first row as identity
    A_bands[0,1] = 0
    A_bands[1,0] = 1
    
    # Right/bottom Neumann boundary condition:
    # Double Endpoints value
    rN = r_ab[-1] # r_beta[Nx]
    A_bands[2,-2] = -2*rN # A[Nx, Nx-1] (A_bands[2,-1] is nonsense due to banded matrix format)
    B[-1,-2] = 2*rN


    C = np.empty((Nt+1, Nx+1)) # (n,i) from (0,0) to (Nt, Nx)
    C[0,:] = 0
    C[0,0] = C0 # Enforce Dirichlet BC at top


    for n in range(Nt):
        RHS = B@C[n] - dt*k*C[n]
        RHS[0]=C0
        C[n+1,:] =la.solve_banded((1,1), A_bands, RHS) # (1,1) denotes A has 1 diagonal row above main diag, 1 below.

    return C

def num_uptake(C0, D, k, L, times, Nx=500, Nt=500):
    '''
    Uses Crank-Nicolson propagation to find concentration profile at a set of final times. 
    Manually recalculates for a series of final times.
    Allows for variable diffusion coeficient D(x).
    Includes explicit reaction term k.
    Uses earlier defined experimental boundary conditions.

    C0: Constant Dirichlet concentration at top boundary.
    D: diffusion coefficient D(x)
    k: reaction rate constant
    L: length of rod
    times: final times to calculate concentration profile at
    Nx: Number of space steps. Including the 0th point, returns a matrix with dimention Nx+1.
    Nt: Number of time steps. Including the 0th point, returns a matrix with dimention Nt+1.
    x_scale: Optional rescale of x grid points.
    '''

    dx = L/Nx # x_0 = 0, x_1 = dx, x_2 = 2*dx, ..., x_i = i*dx, x_Nx = Nx*dx = L

    xgrid = np.linspace(0, L, Nx+1)
    Di = D(xgrid) # Discretizes D(x) to a vector of Nx+1 elements

    uptakes = []
    #for tf in tqdm(times):
    for tf in times:

        dt = tf/Nt # t_0 = 0, t_1 = dt, t_2 = 2*dt, ...,t_n = n*dt, t_Nt = Nt*dt = tf
        
        # Initialize diffusion ratios
        # r_alpha, r_beta differ by 1 index/1 element at each end.
        # r_ab[0] = 1/2 * (D0+D1)*dt / 2*dx^2 = r_alpha[0] = r_beta[1]
        r_ab = 0.5*(Di[:-1] + Di[1:])*dt / (2*dx**2) # Size (Nx,) 
        
        #if k != 0 and dt > 2/k:
        #    tqdm.write(f"dt={dt:.5f} outside of stability conditions! (Nt={Nt}, k={k})")     

        A_bands = np.zeros((3,Nx+1))
        A_bands[0,1:] = -r_ab # Upper Diagonal: -r_alpha[0], -r_alpha[1], -r_alpha[2], ..., -r_alpha[Nx]
        A_bands[1,1:-1] = (1+r_ab[1:]+r_ab[:-1]) # Main Diagonal[i]: 1 + r_alpha[i] + r_alpha[i-1] = 1 + r_alpha[i] + r_beta[i]
        A_bands[1,0] = 1+2*r_ab[0] # Main diagonal endpoints (double r for conservation via ghost points)
        A_bands[1,-1] = 1+2*r_ab[-1]
        A_bands[2,:-1] = -r_ab # Lower Diagonal

        B_bands = np.zeros((3,Nx+1))
        B_bands[0,1:] = r_ab
        B_bands[1,1:-1] = (1-r_ab[1:]-r_ab[:-1])
        B_bands[1,0] = 1-2*r_ab[0]
        B_bands[1,-1] = 1-2*r_ab[-1]
        B_bands[2,:-1] = r_ab

        B = np.diag(B_bands[1,:]) + np.diag(B_bands[0,1:], k=+1) + np.diag(B_bands[2,:-1], k=-1)

        # Lext/top Dirichlet boundary condition:
        # Set first row as identity
        A_bands[0,1] = 0
        A_bands[1,0] = 1
        
        # Right/bottom Neumann boundary condition:
        # Double Endpoints value
        rN = r_ab[-1] # r_beta[Nx]
        A_bands[2,-2] = -2*rN # A[Nx, Nx-1] (A_bands[2,-1] is nonsense due to banded matrix format)
        B[-1,-2] = 2*rN


        C = np.zeros(Nt+1) # Manually rewrites Ci(t) for each x_i from 0 to x_Nx.
        C[0] = C0 # Enforce Dirichlet BC at top

        for n in range(Nt):
            RHS = B@C - dt*k*C
            RHS[0] = C0
            C = la.solve_banded((1,1), A_bands, RHS) # (1,1) denotes A has 1 diagonal row above main diag, 1 below.

        uptakes.append( np.sum(C*dx) ) # [ng/cm^2]

    return uptakes

#-------------Interface: Solve Button and Slider--------------------

def solve():
    print('Solving...')
    global num_frames, C_profile, C_line 

    C0 = float(C0_entrybox.get())
    L  = float(L_entrybox.get())
    tf = float(tf_entrybox.get())

    if L <= 0 or tf <= 0 or C0 <= 0:
        raise ValueError("L, tf, and C0 must be positive values.")
    
    D = make_spline(weights)

    print('Calculating concentration profile...')
    C_profile = num_grid(C0, D, k=0.0, L=L, tf=tf, Nx=500, Nt=500)

    active_frame = 0
    num_frames = C_profile.shape[0]

    print('Plotting...')
    ax.set_xlim(0, L)
    ax.set_ylim(-0.1, C0*1.1)
    C_line, = ax.plot(x, C_profile[active_frame,:], color='blue', label='Concentration Profile C(x,tf)')
    canvas.draw()
    
    print('Done!')


def update_C(slider_value):
    global C_profile, C_line, num_frames

    if 'C_profile' not in globals() or C_line is None:
        return
    
    slider_value = float(slider_value)

    active_frame = min(int(slider_value * (num_frames - 1)), num_frames - 1) # Prevents over-by-one index for final frame

    C_line.set_ydata(C_profile[active_frame, :])

    canvas.draw()


# Solve button
solve_button = ttk.Button(control_panel, text="Solve", command=solve)
solve_button.grid(column=0, row=len(weights)+2, columnspan=3, pady=10)

# Concentration time slider

time_slider = ttk.Scale(control_panel, from_=0, to=1, orient='horizontal', command=update_C)
time_slider.grid(column=0, row=len(weights)+3, columnspan=3, pady=10)
slider.set(0)






root.mainloop()