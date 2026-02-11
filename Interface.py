# Tidied version of SplineInterface.py

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

from utils import make_spline, num_grid, num_uptake

#--------------------------------Initialize D(x)-----------------------------------------------------

Nx = 500
Nt = 500
theta = 0.5
num_weights = 10
weights = np.random.rand(num_weights)*10
L_A = 250 # [A]
L_cm = L_A*1e-8 # [cm] --A is input as angstrom but converted to cm in the code.
alpha = 0.0
beta = 1.0
lin_power = -12
lin_weighting = 1.0
lin_weighting_scaled = lin_weighting*10**lin_power
exp_weighting = 1.0


f = make_spline(weights, L_cm, lin_weighting=lin_weighting_scaled, exp_weighting=exp_weighting)

x = np.linspace(0, L_cm, Nx+1) # For the time being, L must be 100 A and Nx=500

#--------------------------------Initialize Window---------------------------------------------------

root = tk.Tk()
def on_close():
    root.quit()
    root.destroy()
    sys.exit()
root.protocol("WM_DELETE_WINDOW", on_close)

frame = ttk.Frame(root, padding="10")
frame.pack(fill="both", expand=True)
frame.columnconfigure(1, weight=1)
frame.rowconfigure(0, weight=1)

#---------------------------------Plots-----------------------------------------

plotting_window = ttk.Frame(frame, padding='10')
plotting_window.grid(column=1, row=0)


# Fig 1
D_fig = plt.figure(1)
D_ax = D_fig.add_subplot(111)
D_ax.set_title('D(x, t) = f(x) + αC^β [cm^2 / s]')
D_ax.set_xlabel('x [cm]')
D_ax.set_ylabel('f(x) [cm^2 / s]')
D_ax.set_ylim(-0.1e-12, 1.5e-12)
D_ax.set_xlim(0, L_cm)
D_line, = D_ax.plot(x, f(x), linestyle='--', color='grey', label='Diffusion Coefficient D(x), t=0')

D_canvas = FigureCanvasTkAgg(D_fig, master=plotting_window)
D_canvas.get_tk_widget().grid(column=0, row=0)

# Fig 2
C_fig = plt.figure(2)
C_ax = C_fig.add_subplot(111)
C_ax.set_title('')
C_ax.set_xlim(0, L_cm)

C_canvas = FigureCanvasTkAgg(C_fig, master=plotting_window)
C_canvas.get_tk_widget().grid(column=0, row=1)

# Fig 3
m_fig = plt.figure(3)
m_ax = m_fig.add_subplot(111)

m_canvas = FigureCanvasTkAgg(m_fig, master=plotting_window)
m_canvas.get_tk_widget().grid(column=1, row=0, rowspan=2)


def set_bounds():
    global D_line, C0, L, tf, alpha, beta, Nx, Nt, theta, lin_weighting_scaled, exp_weighting, f, x

    C0 = float(C0_entrybox.get())
    L_A  = float(L_entrybox.get())
    tf = float(tf_entrybox.get())
    alpha = float(alpha_entrybox.get())
    beta = float(beta_entrybox.get())
    Nx = int(Nx_entrybox.get())
    Nt = int(Nt_entrybox.get())
    theta = float(theta_entrybox.get())

    if L_A <= 0 or tf <= 0 or C0 <= 0 or Nx <= 0 or Nt <= 0:
        raise ValueError("All entries must be positive values.")
    
    L_cm = L_A*1e-8

    f = make_spline(weights, L_cm, lin_weighting=lin_weighting_scaled, exp_weighting=exp_weighting)
    x = np.linspace(0, L_cm, Nx+1)


    D_ax.set_xlim(0, L_cm)
    C_ax.set_xlim(0, L_cm)
    D_ax.set_ylim(-0.1*np.max(f(x)), 1.1*np.max(f(x)))
    C_ax.set_ylim(-0.1*C0, 1.1*C0)
    
    D_line.set_data(x, f(x))
    D_canvas.draw()


def update_D_weights(slider_num, slider_value):
    global D_line

    weights[slider_num] = float(slider_value)
    f = make_spline(weights, L_cm, lin_weighting=lin_weighting_scaled, exp_weighting=exp_weighting)
    D_line.set_ydata(f(x))
    D_canvas.draw_idle()

    # Update the label
    _, label = weight_sliders[slider_num]
    label.config(text=f"{float(slider_value):.2f}")

def update_D_lin_power(slider_value):
    global D_line, lin_power, lin_weighting, lin_weighting_scaled
    lin_power = np.round(float(slider_value))
    lin_weighting_scaled = lin_weighting*10**lin_power

    f = make_spline(weights, L_cm, lin_weighting=lin_weighting_scaled, exp_weighting=exp_weighting)
    D_line.set_ydata(f(x))
    D_canvas.draw_idle()
    lin_power_value_label.config(text=f"10^{lin_power}")
    lin_weighting_value_label.config(text=f"{lin_weighting:.2f} x 10^{lin_power} cm²/s")
    
def update_D_lin_weighting(slider_value):
    global D_line, lin_weighting, lin_weighting_scaled
    lin_weighting = float(slider_value)
    lin_weighting_scaled = lin_weighting*10**lin_power
    f = make_spline(weights, L_cm, lin_weighting=lin_weighting_scaled, exp_weighting=exp_weighting)
    D_line.set_ydata(f(x))
    D_canvas.draw_idle()
    lin_weighting_value_label.config(text=f"{lin_weighting:.2f} x 10^{lin_power} cm²/s")

def update_D_exp_weighting(slider_value):
    global D_line, exp_weighting
    exp_weighting = float(slider_value)
    f = make_spline(weights, L_cm, lin_weighting=lin_weighting_scaled, exp_weighting=exp_weighting)
    D_line.set_ydata(f(x))
    D_canvas.draw_idle()
    exp_value_label.config(text=f"{exp_weighting:.2f}")


def solve():
    print('Solving...')
    global num_frames, C_profile, C_line, uptake_data, tracker_dot, tracker_line

    set_bounds() 

    print('Calculating concentration profile...')
    C_profile = num_grid(C0, f, k=0.0, L=L_cm, tf=tf, alpha=alpha, beta=beta, Nx=Nx, Nt=Nt, theta=theta)

    active_frame = 0
    num_frames = C_profile.shape[0]

    print('Plotting...')
    for line in C_ax.lines: line.remove()
    C_line, = C_ax.plot(x, C_profile[active_frame,:], color='blue', label='Concentration Profile C(x,tf)')
    C_canvas.draw()
    
    print('Done!')

    # print('Weights: ')
    # print(weights)

    # print('C[-1,:]: ')
    # print(C_profile[-1,:])

    # Plot uptake
    times = np.linspace(0, tf, Nt+1)
    uptake_data = num_uptake(C0, f, k=0.0, L=L_cm, times=times, alpha=alpha, beta=beta, Nx=Nx, Nt=Nt)

    # print('Uptake Data: ')
    # print(uptake_data)

    for line in m_ax.lines: line.remove()
    m_ax.set_xlim(0, tf)
    m_ax.set_ylim(0, 1.1*np.max(uptake_data))
    m_ax.plot(times, uptake_data)
 
    # Create tracking dot
    tracker_dot, = m_ax.plot(
    [0], [uptake_data[0]],
    marker='s',          
    linestyle='None',
    markerfacecolor='none',   
    markeredgecolor='grey',   
    markeredgewidth=1.5,
    markersize=7
    )
    tracker_line = m_ax.axvline(x=0, color='grey', linestyle=':')


    m_canvas.draw()

def update_C(slider_value):
    global C_profile, C_line, num_frames, tf, uptake_data

    if 'C_profile' not in globals() or C_line is None:
        return
    
    
    slider_value = float(slider_value)
    n = int(Nt*slider_value)
    dt = tf / Nt


    C_line.set_ydata(C_profile[n, :])

    C_canvas.draw_idle()


    # Track along uptake curve
    tracker_dot.set_data([n*dt], [uptake_data[n]])
    tracker_line.set_xdata([n*dt])
    m_canvas.draw_idle()

    # Update label
    time_label.config(text=f't = {n*dt:.2f} s')

#---------------------------------Control Panel-----------------------------------------------------

control_panel = ttk.LabelFrame(frame, text="Control Panel", padding="10")
control_panel.grid(column=0, row=0)

#vvvvvvvvvvvvvvv Solver Inputs vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

input_box = ttk.LabelFrame(control_panel, text="Initial Conditions", padding="10")
input_box.pack(fill='x', pady=5)


# C0 [ng/cm^3]
C0_value = tk.DoubleVar(value=1e9) # Density of water
# Right label (units)
C0_unit = ttk.Label(input_box, text="ng/cm³")
C0_unit.grid(row=0, column=2, padx=5)
# Left label (variable name)
C0_label = ttk.Label(input_box, text="C0: ")
C0_label.grid(row=0, column=0, padx=5)
# Entry (float)
C0_entrybox = ttk.Entry(input_box, textvariable=C0_value, width=10, justify="right")
C0_entrybox.grid(row=0, column=1, padx=5)


# L [A]
L_value = tk.DoubleVar(value=250.0)
# Right label (units)
L_unit = ttk.Label(input_box, text="Å")
L_unit.grid(row=1, column=2, padx=5)
# Left label (variable name)
L_label = ttk.Label(input_box, text="L: ")
L_label.grid(row=1, column=0, padx=5)
# Entry (float)
L_entrybox = ttk.Entry(input_box, textvariable=L_value, width=10, justify="right")
L_entrybox.grid(row=1, column=1, padx=5)


# tf [s]
tf_value = tk.DoubleVar(value=10.0)
# Right label (units)
tf_unit = ttk.Label(input_box, text="s")
tf_unit.grid(row=2, column=2, padx=5)
# Left label (variable name)
tf_label = ttk.Label(input_box, text="tf: ")
tf_label.grid(row=2, column=0, padx=5)
# Entry (float)
tf_entrybox = ttk.Entry(input_box, textvariable=tf_value, width=10, justify="right")
tf_entrybox.grid(row=2, column=1, padx=5)

# alpha
alpha_value = tk.DoubleVar(value=0.0)
# Label
alpha_label = ttk.Label(input_box, text='alpha: ')
alpha_label.grid(row=3, column=0, padx=5)
# Entry (float)
alpha_entrybox = ttk.Entry(input_box, textvariable=alpha_value, width=10, justify='right')
alpha_entrybox.grid(row=3, column=1, padx=5)

# beta
beta_value = tk.DoubleVar(value=1.0)
# Label
beta_label = ttk.Label(input_box, text='beta: ')
beta_label.grid(row=4, column=0, padx=5)
# Entry (float)
beta_entrybox = ttk.Entry(input_box, textvariable=beta_value, width=10, justify='right')
beta_entrybox.grid(row=4, column=1, padx=5)

# Nx
Nx_value = tk.IntVar(value=Nx)
# Label (variable name)
Nx_label = ttk.Label(input_box, text='Nx: ')
Nx_label.grid(row=0, column=4, padx=5)
# Entry (int)
Nx_entrybox = ttk.Entry(input_box, textvariable=Nx_value, width=10, justify='right')
Nx_entrybox.grid(row=0, column=5)

# Nt
Nt_value = tk.IntVar(value=Nt)
# Label (variable name)
Nt_label = ttk.Label(input_box, text='Nt: ')
Nt_label.grid(row=1, column=4, padx=5)
# Entry (int)
Nt_entrybox = ttk.Entry(input_box, textvariable=Nt_value, width=10, justify='right')
Nt_entrybox.grid(row=1, column=5)

# Theta
theta_value = tk.DoubleVar(value=theta)
# Label
theta_label = ttk.Label(input_box, text='Theta: ')
theta_label.grid(row=2, column=4, padx=5)
# Entry (float)
theta_entrybox = ttk.Entry(input_box, textvariable=theta_value, width=10, justify='right')
theta_entrybox.grid(row=2, column=5)


# Set bounds button
set_bounds_button = ttk.Button(input_box, text="Set Bounds", command=set_bounds)
set_bounds_button.grid(column=0, row=5, columnspan=3, pady=2)

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#vvvvvvvvvvvvvvvvvvv D Weighting vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

weighting_box =  ttk.LabelFrame(control_panel, text="Weighting", padding="10")
weighting_box.pack(fill='x', pady=5)


# Linear multiplier slider:
# Value
lin_power_value_label = ttk.Label(weighting_box, text=f"10^{lin_power}")
lin_power_value_label.grid(column=2, row=0)
# Label
lin_power_label = ttk.Label(weighting_box, text='Order of Magnitude: ')
lin_power_label.grid(column=0, row=0)
# Slider 
lin_power_slider = ttk.Scale(weighting_box, from_=-20, to=-5, orient='horizontal', command=update_D_lin_power) # from 1e-20 to 1e-5
lin_power_slider.set(lin_power) # 1e-12
lin_power_slider.grid(column=1, row=0)


# Linear adjustment slider:
# Value
lin_weighting_value_label = ttk.Label(weighting_box, text=f"{lin_weighting:.2f} x 10^{lin_power} cm²/s")
lin_weighting_value_label.grid(column=2, row=1)
# Label
lin_weighting_label = ttk.Label(weighting_box, text='Linear: ')
lin_weighting_label.grid(column=0, row=1)
# Slider 
lin_weighting_slider = ttk.Scale(weighting_box, from_=1, to=10, orient='horizontal', command=update_D_lin_weighting) # from 1e-13 to 20e-13 aka 2e-12
lin_weighting_slider.set(lin_weighting) # 10e-13 = 1e-12
lin_weighting_slider.grid(column=1, row=1)



# Exponential slider:
# Value
exp_value_label = ttk.Label(weighting_box, text=f"{exp_weighting:.2f}")
exp_value_label.grid(column=2, row=2)
# Label
exp_label = ttk.Label(weighting_box, text='Exponential: ')
exp_label.grid(column=0, row=2)
# Slider
exp_slider = ttk.Scale(weighting_box, from_=0, to=10, orient='horizontal', command=update_D_exp_weighting)
exp_slider.set(exp_weighting)
exp_slider.grid(column=1, row=2)



#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


#vvvvvvvvvvvvvvvvvvvvvvv D Weights vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

weight_box = ttk.LabelFrame(control_panel, text="Weights", padding="10")
weight_box.pack(fill='x', pady=5)

weight_sliders = {}
for n in range(num_weights):
    # Create label
    slider_label = ttk.Label(weight_box, text=f"Weight {n}: ")
    slider_label.grid(column=0, row=n)
    # Show value
    value_label = ttk.Label(weight_box, text=f"{weights[n]:.2f}")
    value_label.grid(column=2, row=n, padx=5)

    # Create Slider
    slider = ttk.Scale(
        weight_box, 
        from_=0, to=10, 
        orient='horizontal', 
        command=lambda v, slider_num=n: update_D_weights(slider_num, v)
    )

    # Store both slider and value label for future updates
    weight_sliders[n] = (slider, value_label)

    # Initialize slider values and grid locations
    slider.set(weights[n])
    slider.grid(column=1, row=n)


    

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#vvvvvvvvvvvvvvvv Solve Button and Solution vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

soln_box = ttk.LabelFrame(control_panel)
soln_box.pack(fill='x', pady=5)

# Button
solve_button = ttk.Button(soln_box, text="Solve", command=solve)
solve_button.grid(column=0, row=0, columnspan=2, pady=10)

# Concentration time slider
time_slider = ttk.Scale(soln_box, from_=0, to=1, orient='horizontal', command=update_C)
time_slider.grid(column=0, row=1, pady=10)
time_slider.set(0)

# Time value label
time_label = ttk.Label(soln_box, text='t = 0.00 s')
time_label.grid(column=1, row=1, pady=10)


root.mainloop()