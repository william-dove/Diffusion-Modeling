import numpy as np
import scipy.linalg as la
from tqdm import tqdm
import jax
import jax.numpy as jnp


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



def num_grid(C0, D, k, L, tf, Nx=500, Nt=500):
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

def num_uptake(C0, D, k, L, times, Nx=500, Nt=500, mode='fast'):
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
    
    if mode == 'fast':
        times_needed = times[-1]
    elif mode == 'accurate':
        times_needed = times
    else:
        raise ValueError('Incorrect mode argument; use "fast" or "times"')

    for tf in tqdm(times_needed):

        dt = tf/Nt # t_0 = 0, t_1 = dt, t_2 = 2*dt, ...,t_n = n*dt, t_Nt = Nt*dt = tf
        
        # Initialize diffusion ratios
        # r_alpha, r_beta differ by 1 index/1 element at each end.
        # r_ab[0] = 1/2 * (D0+D1)*dt / 2*dx^2 = r_alpha[0] = r_beta[1]
        r_ab = 0.5*(Di[:-1] + Di[1:])*dt / (2*dx**2) # Size (Nx,) 
        
        if k != 0 and dt > 2/k:
            tqdm.write(f"Warning: dt={dt:.5f} outside of stability conditions! (Nt={Nt}, k={k})")     

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