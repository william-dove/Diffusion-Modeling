import numpy as np
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

#------------------------JAX and Error Calculation Stuff--------------------------------------------------------

def L2norm(num_soln, an_soln):
    '''
    Computes the distance between "points" in the solution vector and analytic solution vector.
    This is normalized by the norm of the analytic solution.  
    '''
    if len(num_soln) != len(an_soln):
        raise ValueError(f"Vectors must be the same length (sizes {len(num_soln)} and {len(an_soln)}).")
    else:
        err = jnp.linalg.norm(num_soln - an_soln) / jnp.linalg.norm(an_soln)
    return err


#----------------------Numerical Solutions----------------------------------------------------------------------

# All numerical soluions using JAX syntax

def make_banded(Di, dx, dt, reaction_mode, k, theta):
    '''  
    Takes in a vector representing the variable diffusion coefficient, and returns A and B matrices in banded form.
    '''

    # Initialize diffusion ratios
    # r_alpha, r_beta differ by 1 index/1 element at each end.
    # r_ab[0] = 1/2 * (D0+D1)*dt / 2*dx^2 = r_alpha[0] = r_beta[1]

    # !!! OLD: New one is below with proper variable CN weighting. !!!
    '''
    r_ab = 0.5*(Di[:-1] + Di[1:])*dt / (2*dx**2) # Size (Nx,)  
    '''
    r_ab = (Di[:-1] + Di[1:])*dt / (2*dx**2) # No more baked-in CN weighting, this is applied below in A,B Construction.


    # Create banded matrices for jnp.lax.linalg.tridiagonal_solve
    # Format: first element of lower diag = 0
    # Format: last element of upper diag  = 0

    # Lower Diagonal: Identical to upper diagonal
    A_lower = jnp.concatenate([
        jnp.array([0]), 
        -theta*r_ab
    ]) 
    # Main Diagonal[i] = 1 + r_alpha[i] + r_alpha[i-1] = 1 + r_alpha[i] + r_beta[i]
    # Main Diagonal: 1+2r_alpha[0], ...,  1 + r_alpha[i] + r_beta[i], ..., 1+2r_alpha[Nx]
    A_main  = jnp.concatenate([
        jnp.array([1 + 2*theta*r_ab[0]]), 
        1 + theta*r_ab[1:] + theta*r_ab[:-1], 
        jnp.array([1 + 2*theta*r_ab[-1]])
    ])
    # Upper Diagonal: -r_alpha[0], -r_alpha[1], -r_alpha[2], ..., -r_alpha[Nx]
    A_upper = jnp.concatenate([
        -theta*r_ab,
        jnp.array([0])
    ]) 
    # B: Reverse signs from A
    B_lower = jnp.concatenate([
        jnp.array([0]),
        (1-theta)*r_ab
    ])
    B_main = jnp.concatenate([
        jnp.array([1 - 2*(1-theta)*r_ab[0]]),
        1 - (1-theta)*r_ab[1:] - (1-theta)*r_ab[:-1],
        jnp.array([1 - 2*(1-theta)*r_ab[-1]])
    ])
    B_upper = jnp.concatenate([
        (1-theta)*r_ab,
        jnp.array([0])
    ])

    # Implicit Reaction:
    if reaction_mode == 'implicit':
        A_main += theta * dt * k
        B_main -= (1 - theta) * dt * k

    # Lext/top Dirichlet boundary condition:
    # Set first row of A and B as identity
    A_upper = A_upper.at[0].set(0)
    A_main = A_main.at[0].set(1)
    B_upper = B_upper.at[0].set(0)
    B_main = B_main.at[0].set(1)
        
    # Right/bottom Neumann boundary condition:
    # Double Endpoints value (on diagonals; main is already doubled above)
    rN = r_ab[-1] # r_beta[Nx]
    A_lower = A_lower.at[-1].set(-2*theta*rN) 
    B_lower = B_lower.at[-1].set(2*(1-theta)*rN)

    return A_lower, A_main, A_upper, B_lower, B_main, B_upper


def num_grid(C0, f, k, L, tf, Nx=50, Nt=6400, alpha=0.0, beta=1.0, theta=0.55, reaction_mode='implicit'):
    '''
    Numerical solution for experimental boundary conditions

    Uses Crank-Nicolson propagation to find concentration profile over time. 
    Allows for variable diffusion coeficient D(x).
    Includes explicit reaction term k.
    Uses earlier defined experimental boundary conditions.

    Default conditions should result in <5% error:
    - theta=0.55, Nx=50, Nt=6400

    Optimized to work with Jax. 
    Only supports implicit reaction mode, would probably be best to keep k=0.
    Only supports D with no C dependance.

    C0: Constant Dirichlet concentration at top boundary.
    D: diffusion coefficient D(x)
    k: reaction rate constant
    L: length of rod
    tf: final time
    Nx: Number of space steps. I am including a 0th point in xgrid, thus returns a matrix with dimention Nx+1.
    Nt: Number of time steps. jax.lax.scan does NOT Include the 0th point, thus returns a matrix with dimention Nt.
    alpha, beta: Experimental parameters to introduce C dependance to D.
    theta: weights the average between implicit and explicit timestepping
    '''
    if reaction_mode != 'implicit':
        raise ValueError('Only suports implicit reaction mode currently.')
    if alpha != 0 or beta != 1:
        raise ValueError('JAX version does not support C-dependant D currently.')

    dx = L/Nx # x_0 = 0, x_1 = dx, x_2 = 2*dx, ..., x_i = i*dx, x_Nx = Nx*dx = L
    dt = tf/Nt # t_0 = 0, t_1 = dt, t_2 = 2*dt, ...,t_n = n*dt, t_Nt = Nt*dt = tf
    xgrid = jnp.linspace(0, L, Nx+1)
    Df = f(xgrid) # f(x) contribution to D(x)
    Di = Df # Discretizes D(x) to a vector of Nx+1 elements
    A_lower, A_main, A_upper, B_lower, B_main, B_upper = make_banded(Di, dx, dt, reaction_mode, k, theta) # Initialize A and B in banded format

    # Initialize Ci(tn)
    Ci = jnp.zeros(Nx+1) # Manually rewrites Ci(tn) for each timestep.
    # Enforce Dirichlet BC at top
    Ci = Ci.at[0].set(C0) 

    # Time loop formatted using jax.lax.scan
    def step(Ci, _):
        # Efficient B@C
        RHS = (
            B_main*Ci +
            B_lower*jnp.roll(Ci, 1) +
            B_upper*jnp.roll(Ci, -1)
        )
        # Enforce Dirichlet BC
        RHS = RHS.at[0].set(C0)
        # Convert to column vector for jax
        RHS_rank2 = RHS[:, None]
        # Set the solution to Ci[n+1], increment n and repeat 
        Ci = tridiagonal_solve(A_lower, A_main, A_upper, RHS_rank2)
        Ci = Ci.flatten()  # Convert back to 1D array

        return Ci, Ci

    _, C_grid = jax.lax.scan(
        f = step,
        init = Ci,
        xs = None,
        length = Nt
    )

    return C_grid


def num_uptake(C0, f, k, L, times, Nx=50, Nt=6400, alpha=0, beta=1, theta=0.55, reaction_mode='implicit'):
    '''
    Uses Crank-Nicolson propagation to find concentration profile at a set of final times. 
    Allows for variable diffusion coeficient D(x).
    Includes explicit reaction term k.
    Uses earlier defined experimental boundary conditions.

    Default conditions should result in <5% error:
    - theta=0.55, Nx=50, Nt=6400

    Optimized to work with Jax. 
    Only supports implicit reaction mode, would probably be best to keep k=0.
    Only supports D with no C dependance.

    C0: Constant Dirichlet concentration at top boundary.
    f: Function giving spatial dependance for D(x) = f(x) + alpha*C(x)^beta [cm^2 / s]
    k: reaction rate constant
    L: length of rod [cm]
    times: final times to calculate concentration profile at
    Nx: Number of space steps used to compute the answer. Including the 0th step, this results in an xgrid of shape (Nx+1,).
    Nt: Number of time steps used to compute the answer. i.e., starting at 0 before taking a step, it then takes 500 step to reach Nt=500.
    alpha: linear coefficient for C(x) dependancy on D(x, C)
    beta: exponential factor for C(x) dependancy on D(x, C)
    theta: Alters CN weighting from pure CN
    '''
    if reaction_mode != 'implicit':
        raise ValueError('Only suports implicit reaction mode currently.')
    if alpha != 0 or beta != 1:
        raise ValueError('JAX version does not support C-dependant D currently.')
    
    tf = jnp.max(times)
    dx = L/Nx # x_0 = 0, x_1 = dx, x_2 = 2*dx, ..., x_i = i*dx, x_Nx = Nx*dx = L
    dt = tf/Nt # t_0 = 0, t_1 = dt, t_2 = 2*dt, ...,t_n = n*dt, t_Nt = Nt*dt = tf
    xgrid = jnp.linspace(0, L, Nx+1)
    Df = f(xgrid) # f(x) contribution to D(x)
    Di = Df # Discretizes D(x) to a vector of Nx+1 elements
    A_lower, A_main, A_upper, B_lower, B_main, B_upper = make_banded(Di, dx, dt, reaction_mode, k, theta) # Initialize A and B in banded format

    # Initialize Ci(tn)
    Ci = jnp.zeros(Nx+1) # Manually rewrites Ci(tn) for each timestep.
    # Enforce Dirichlet BC at top
    Ci = Ci.at[0].set(C0) 

    # Time loop formatted using jax.lax.scan
    def step(Ci, _):
        # Efficient B@C
        RHS = (
            B_main*Ci +
            B_lower*jnp.roll(Ci, 1) +
            B_upper*jnp.roll(Ci, -1)
        )
        # Enforce Dirichlet BC
        RHS = RHS.at[0].set(C0)
        # Convert to column vector for jax
        RHS_rank2 = RHS[:, None]
        # Set the solution to Ci[n+1], increment n and repeat 
        Ci = tridiagonal_solve(A_lower, A_main, A_upper, RHS_rank2)
        Ci = Ci.flatten()  # Convert back to 1D array

        # Integrate the new uptake
        uptake = jnp.sum(Ci*dx)

        return Ci, uptake

    Cf, uptakes = jax.lax.scan(
        f = step,
        init = Ci,
        xs = None,
        length = Nt
    )


    # Interpolate the concentration profile data to the given times.
    tgrid = jnp.linspace(dt, tf, Nt) # tgrid = [1dt, 2dt, 3dt, ..., Nt*dt=tf]
    uptakes = jnp.interp(times, tgrid, uptakes)

    return uptakes

def accurate_num_uptake(C0, f, k, L, times, Nx=500, Nt=500, alpha=0, beta=1, theta=0.5, reaction_mode='implicit'):
    '''
    Uses Crank-Nicolson propagation to find concentration profile at a set of final times. 
    Manually recalculates for a series of final times.
    Allows for variable diffusion coeficient D(x).
    Includes explicit reaction term k.
    Uses earlier defined experimental boundary conditions.

    DOES NOT WORK WITH JAX

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



    uptakes = []
    for tf in tqdm(times, desc='Exact Points Calculated'):

        dt = tf/Nt # t_0 = 0, t_1 = dt, t_2 = 2*dt, ...,t_n = n*dt, t_Nt = Nt*dt = tf
        
        if k != 0 and dt > 2/k:
            tqdm.write(f"Warning: dt={dt:.5f} outside of stability conditions! (Nt={Nt}, k={k})")    

        A_lower, A_main, A_upper, B_lower, B_main, B_upper = make_banded(Di, dx, dt, reaction_mode, k, theta)


        # Initialize
        C = jnp.zeros(Nx+1) # Manually rewrites Ci(t) for each x_i from 0 to x_Nx.
        # Enforce Dirichlet BC at top
        C = C.at[0].set(C0) 
        
        # Soln loop
        for n in range(Nt):
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
                A_lower, A_main, A_upper, B_lower, B_main, B_upper = make_banded(Di, dx, dt, reaction_mode, k, theta)

        # Since the concentration profile data is not stored, just append the current uptake at tf and repeat for a new tf.
        uptakes.append( jnp.sum(C*dx) ) # [ng/cm^2]

    return uptakes


#-------------------Analytic Solutions-----------------------------------

def an_grid(C0, D, L, tf, Nx=500, Nt=500, M=50):
    '''
    Computes the analytic solution to the diffusion equation for the given boundary and initial conditions.
    Returns solution as an array of format C[t, x] and size (Nt+1, Nx+1).
    '''
    dt = tf/Nt
    t = jnp.linspace(dt, tf, Nt)
    x = jnp.linspace(0, L, Nx+1)
    
    T, X = jnp.meshgrid(t,x, indexing='ij')

    C = jnp.full((Nt, Nx+1), C0, dtype=jnp.float32)

    def step(C, m):
        lambda_m = ( (m+0.5)*jnp.pi / L )**2
        a_m = -(2*C0) / (L*jnp.sqrt(lambda_m))
        C += a_m*jnp.sin(jnp.sqrt(lambda_m)*X)*jnp.exp(-D*lambda_m*T)
        return C, None
    
    C_final, _ = jax.lax.scan(
        step,
        C,
        jnp.arange(M)
    )

    # Deprecated:
    '''
    for m in jnp.arange(M):
        lambda_m = ( (m+0.5)*jnp.pi / L )**2
        a_m = -(2*C0) / (L*jnp.sqrt(lambda_m))
        C += a_m*jnp.sin(jnp.sqrt(lambda_m)*X)*jnp.exp(-D*lambda_m*T)
    '''
    return C_final

def an_uptake(C0, D, L, times, Nx=500, M=500):
    '''
    Computes the analytic solution to the diffusion equation for the given boundary and initial conditions.
    Computes the mass uptake at all times in the ijnput vector t for comparison.

    C0: Initial concentration of the solvent drop [ng/cm^3]
    D: Diffusion coefficient [cm^2/s]
    L: Film thickness [cm]
    times: different time points [s]
    '''
    
    x = jnp.linspace(0, L, Nx+1) # [cm]
    dx = L/Nx  # [cm]

    uptakes = []

    for t in tqdm(times):

        C = jnp.full(Nx+1, C0, dtype='float')

        for m in jnp.arange(M):
            lambda_m = ( (m+0.5)*jnp.pi / L )**2
            a_m = -(2*C0) / (L*jnp.sqrt(lambda_m))
            C += a_m*jnp.sin(jnp.sqrt(lambda_m)*x)*jnp.exp(-D*lambda_m*t)

        uptakes.append( jnp.sum(C*dx) ) # [ng/cm^2]

    return uptakes


def C_uptake(C, L=1, tf=1):
    '''
    Calculates the mass uptake over time for concentration profile data.
    Integrates concentration profile to get total solute for each time.
    '''

    Nx = C.shape[1]
    Nt = C.shape[0]
    dx = L/Nx
    dt = tf/Nt

    uptake_data = []

    for i in range(C.shape[0]):
        t = i*dt
        m = jnp.sum(C[i]*dx)

        uptake_data.append([t, m])

    return jnp.array(uptake_data).T