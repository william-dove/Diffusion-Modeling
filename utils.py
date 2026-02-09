import numpy as np
import scipy.linalg as la
from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax.lax.linalg import tridiagonal_solve
from jax.scipy.interpolate import RegularGridInterpolator


#-----------------Spline Stuff------------------------------------------------------------------------
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

def make_spline(weights, L, lin_weighting=1e-12, exp_weighting=1.0, median=0.0):
    '''  
    Makes a spline with the number of knots/nodes determined by how many weights are given.
    Expects weights in an array, with values from 1 to 10

    weights: array of shape (n,), containing values between 0 and 1
    L: length of the interval [0, L]
    
    Sums the piecewise basis functions defined in cubic_bspline, scaled by the weights.
    '''

    weights = jnp.asarray(weights)
    n = weights.shape[0]
    dx = L / (n - 1)
    x_nodes = jnp.linspace(0, L, n)

    adjusted_weights = median + (weights**exp_weighting) * lin_weighting
        

    def spline(x):
        u = (x[..., None] - x_nodes) / dx
        return jnp.sum(adjusted_weights * cubic_bspline(u), axis=-1) / 10 # Undoes the weights being from 1 to 10

    return jax.jit(spline)

#----------------------Numerical Solutions----------------------------------------------------------------------

# All numerical soluions using JAX syntax

def make_banded(Di, dx, dt, reaction_mode, k):
    '''  
    Takes in a vector representing the variable diffusion coefficient, and returns A and B matrices in banded form.
    '''

    # Initialize diffusion ratios
    # r_alpha, r_beta differ by 1 index/1 element at each end.
    # r_ab[0] = 1/2 * (D0+D1)*dt / 2*dx^2 = r_alpha[0] = r_beta[1]
    r_ab = 0.5*(Di[:-1] + Di[1:])*dt / (2*dx**2) # Size (Nx,)  


    # Create banded matrices for jnp.lax.linalg.tridiagonal_solve
    # Format: first element of lower diag = 0
    # Format: last element of upper diag  = 0

    # Lower Diagonal: Identical to upper diagonal
    A_lower = jnp.concatenate([
        jnp.array([0]), 
        -r_ab
    ]) 
    # Main Diagonal[i] = 1 + r_alpha[i] + r_alpha[i-1] = 1 + r_alpha[i] + r_beta[i]
    # Main Diagonal: 1+2r_alpha[0], ...,  1 + r_alpha[i] + r_beta[i], ..., 1+2r_alpha[Nx]
    A_main  = jnp.concatenate([
        jnp.array([1+2*r_ab[0]]), 
        1+r_ab[1:]+r_ab[:-1], 
        jnp.array([1+2*r_ab[-1]])
    ])
    # Upper Diagonal: -r_alpha[0], -r_alpha[1], -r_alpha[2], ..., -r_alpha[Nx]
    A_upper = jnp.concatenate([
        -r_ab,
        jnp.array([0])
    ]) 
    # B: Reverse signs from A
    B_lower = jnp.concatenate([
        jnp.array([0]),
        r_ab
    ])
    B_main = jnp.concatenate([
        jnp.array([1-2*r_ab[0]]),
        1-r_ab[1:]-r_ab[:-1],
        jnp.array([1-2*r_ab[-1]])
    ])
    B_upper = jnp.concatenate([
        r_ab,
        jnp.array([0])
    ])

    # Implicit Reaction:
    if reaction_mode == 'implicit':
        A_main += dt*k/2
        B_main -= dt*k/2

    # Lext/top Dirichlet boundary condition:
    # Set first row of A and B as identity
    A_upper = A_upper.at[0].set(0)
    A_main = A_main.at[0].set(1)
    B_upper = B_upper.at[0].set(0)
    B_main = B_main.at[0].set(1)
        
    '''
    # Right/bottom Neumann boundary condition:
    # Double Endpoints value
    rN = r_ab[-1] # r_beta[Nx]
    A_bands[2,-2] = -2*rN # A[Nx, Nx-1] (A_bands[2,-1] is nonsense due to banded matrix format)
    B[-1,-2] = 2*rN
    '''

    return A_lower, A_main, A_upper, B_lower, B_main, B_upper


def num_grid(C0, f, k, L, tf, Nx=500, Nt=500, alpha=0.0, beta=1.0, reaction_mode='implicit'):
    '''
    Numerical solution for experimental boundary conditions

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

    if k != 0 and dt > 2/k:
        tqdm.write(f"Warning: dt={dt:.5f} outside of stability conditions! (Nt={Nt}, k={k})")    

    xgrid = jnp.linspace(0, L, Nx+1)
    Df = f(xgrid)
    Di = Df # Discretizes D(x) to a vector of Nx+1 elements

    A_lower, A_main, A_upper, B_lower, B_main, B_upper = make_banded(Di, dx, dt, reaction_mode, k)

    # Initialize C grid
    C = jnp.zeros((Nt+1, Nx+1))
    C = C.at[0,0].set(C0)


    for n in tqdm(range(Nt)):

        Cn = C[n,:]
        # Efficient B@C
        RHS = (
            B_main*Cn +
            B_lower*jnp.roll(Cn, 1) +
            B_upper*jnp.roll(Cn, -1)
        )
        # Correct for reaction term k
        if reaction_mode == 'explicit':
            RHS -= dt*k*Cn
        # Enforce Dirichlet BC
        RHS = RHS.at[0].set(C0)
        # Convert to column vector for jax
        RHS_rank2 = RHS[:, None]
        # Set the solution to C[n+1], increment n and repeat 
        Cnp1 = tridiagonal_solve(A_lower, A_main, A_upper, RHS_rank2)
        Cnp1 = Cnp1.flatten()  # Convert back to 1D array

        # Account for D dependance on C
        if alpha != 0:
            Di = Df + alpha*Cnp1**beta
            A_lower, A_main, A_upper, B_lower, B_main, B_upper = make_banded(Di, dx, dt, reaction_mode, k)

        # Write to new C[n]
        C = C.at[n+1, :].set(Cnp1)

    return C


def num_uptake(C0, f, k, L, times, Nx=500, Nt=500, alpha=0, beta=1, mode='fast', reaction_mode='implicit'):
    '''
    Uses Crank-Nicolson propagation to find concentration profile at a set of final times. 
    Manually recalculates for a series of final times.
    Allows for variable diffusion coeficient D(x).
    Includes explicit reaction term k.
    Uses earlier defined experimental boundary conditions.

    C0: Constant Dirichlet concentration at top boundary.
    f: Function giving spatial dependance for D(x) = f(x) + alpha*C(x)^beta [cm^2 / s]
    k: reaction rate constant
    L: length of rod [cm]
    times: final times to calculate concentration profile at
    Nx: Number of space steps. Including the 0th point, returns a matrix with dimention Nx+1.
    Nt: Number of time steps. Including the 0th point, returns a matrix with dimention Nt+1.
    alpha: linear coefficient for C(x) dependancy on D(x, C)
    beta: exponential factor for C(x) dependancy on D(x, C)
    '''

    dx = L/Nx # x_0 = 0, x_1 = dx, x_2 = 2*dx, ..., x_i = i*dx, x_Nx = Nx*dx = L

    xgrid = jnp.linspace(0, L, Nx+1)
    Df = f(xgrid) # f(x) contribution to D(x)
    Di = Df # Discretizes D(x) to a vector of Nx+1 elements

    
    if mode == 'fast':
        times_needed = [max(times)]
        times_iter = times_needed
    elif mode == 'accurate':
        times_needed = times
        times_iter = tqdm(times_needed, desc='Exact Points Calculated')
    else:
        raise ValueError('Incorrect mode argument; use "fast" or "times"')

    uptakes = []
    for tf in times_iter:

        dt = tf/Nt # t_0 = 0, t_1 = dt, t_2 = 2*dt, ...,t_n = n*dt, t_Nt = Nt*dt = tf
        
        if k != 0 and dt > 2/k:
            tqdm.write(f"Warning: dt={dt:.5f} outside of stability conditions! (Nt={Nt}, k={k})")    

        A_lower, A_main, A_upper, B_lower, B_main, B_upper = make_banded(Di, dx, dt, reaction_mode, k)


        # Initialize
        C = jnp.zeros(Nx+1) # Manually rewrites Ci(t) for each x_i from 0 to x_Nx.
        # Enforce Dirichlet BC at top
        C = C.at[0].set(C0) 

        if mode == 'fast': # Store a single C profile grid to interpolate later.
            C_grid = jnp.zeros((Nt+1, Nx+1))
            C_grid = C_grid.at[0,:].set(C)
        
        # Soln loop
        for n in tqdm(range(Nt)):
            # Efficient B@C
            RHS = (
                B_main*C +
                B_lower*jnp.roll(C, 1) +
                B_upper*jnp.roll(C, -1)
            )
            # Correct for reaction term k
            if reaction_mode == 'explicit':
                RHS -= dt*k*C
            # Enforce Dirichlet BC
            RHS = RHS.at[0].set(C0)
            # Convert to column vector for jax
            RHS_rank2 = RHS[:, None]
            # Set the solution to C[n+1], increment n and repeat 
            C = tridiagonal_solve(A_lower, A_main, A_upper, RHS_rank2)
            C = C.flatten()  # Convert back to 1D array

            # Account for D dependance on C
            if alpha != 0:
                Di = Df + alpha*C**beta
                A_lower, A_main, A_upper, B_lower, B_main, B_upper = make_banded(Di, dx, dt, reaction_mode, k)

            if mode == 'fast': # Store the concentration profile data from t=0 to t=max(times), to be interpolated later
                C_grid = C_grid.at[n+1].set(C)

        if mode == 'accurate': # Since the concentration profile data is not stored, just append the current uptake at tf and repeat for a new tf.
            uptakes.append( np.sum(C*dx) ) # [ng/cm^2]
        elif mode == 'fast': # Interpolate the concentration profile data to the given times.
            tgrid = jnp.linspace(0, tf, Nt+1)
            interpolate = RegularGridInterpolator(
                points=(tgrid, xgrid),
                values=C_grid,
                method='linear'
            )
            for t in times:
                C = interpolate((t, xgrid))
                uptakes.append( np.sum(C*dx) )

    return uptakes