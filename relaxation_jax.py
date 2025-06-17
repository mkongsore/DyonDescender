# Load packages
import numpy as np
import scipy
from scipy.integrate import solve_bvp
from scipy.ndimage import shift
import itertools
import sys
from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import fori_loop
from functools import partial

job_idx = int(sys.argv[1]) # Pick out job index from job array
R=0.1*float(job_idx) # Dyon loop radius [1/vev]

spacing = 0.1 # Grid spacing [1/vev]
dx2 = spacing**2 # Square spacing [1/vev^2]
volume = spacing**4 # Volume per grid point [1/vev^4]
length = R+1.55  # Grid extends from -length to length in all four directions [1/vev]

grid = np.arange(-length,length+spacing,spacing) # 1D grid from which 4D grid is constructed
grid2 = np.arange(-(length+spacing/2),length+spacing/2,spacing)

X,Y,Z,T = np.meshgrid(grid,grid,grid2,grid2,indexing='ij') # 4d grid in x,y,z,t

slice_idx = int(np.around(np.size(grid)/2.)) # index corresponding to slice through loop center

# Solve the Georgi-Glashow field equations that yield the Higgs and gauge field profile function
# for a static 't Hooft-Polyakov monopole -- see e.g. Shifman for equations.

r = np.linspace(0.0001,10,10000) # Test radii [1/vev]
lamb = 0.5 # The Higgs quartic coupling [no unit]
g = 1.0 # The SU(2) gauge coupling [no unit]
vev = 1.0 # The Higgs vev [vev]

l = 1

def fun(r,w):
    """
    Defines the differential equations for the h and k Higgs functions, given in Shifman.

    Arguments:
        w :  vector of the state variables:
                  w = [hp,kp,h,k]
        t :  time
        p :  vector of the parameters:
                  p = [lambda]
    """

    f = [-2/r*w[0]+2/r**2*w[3]**2*w[2]+lamb*(w[2]**2-1)*w[2],
         1/r**2*(w[3]**2-1)*w[3]+w[2]**2*w[3],
         w[0],
         w[1]]

    return np.vstack(f)

def bc(wa,wb):
    return np.array([wa[2],wa[3]-1.,wb[2]-1.,wb[3]])

# Provide some initial guesses for solving the profile equations
guess = np.ones((4,np.size(r)))

guess[0,:]=0
guess[1,:]=0

guess[2,0]=0
guess[2,-1]=1
guess[3,0]=1
guess[3,-1]=0

# Solve the profile equations
res = solve_bvp(fun, bc, r, guess,tol=1e-10)

# Define three distinct initial guess profile functions
def h_fun(r):
    return res.sol(r)[2]
def k_fun(r):
    return res.sol(r)[3]
def ken_fun(r2):
    val = 1 - np.exp(-r2 / 2)
    val = np.where(r2 > 1, 1.0, val)
    return val

def levi_cevita_tensor(dim): # First, define levi civita tensor with (dim) number of indices
    arr=np.zeros(tuple([dim for _ in range(dim)]))
    for x in itertools.permutations(tuple(range(dim))):
        mat = np.zeros((dim, dim), dtype=np.int32)
        for i, j in zip(range(dim), x):
            mat[i, j] = -1
        arr[x]=-int(np.linalg.det(mat))
    return arr

# Define double polar coordainte system u,v,phi,tau
def phi(x,y):
    return np.arctan2(y,x) # [rad]

def tau(z,t):
    return np.arctan2(t,z) # [rad]

def u(x,y):
    return np.sqrt(x**2+y**2) # [1/vev]

def v(z,t):
    return np.sqrt(z**2+t**2) # [1/vev]

# Define beta angle
def beta(x,y,z,t):
    return np.arctan2(u(x,y),v(z,t)-R)+np.arctan2(u(x,y),v(z,t)+R) # [rad]

# Set up guess Higgs field configuration
Hhat=np.array([np.sin(beta(X,Y,Z,T))*np.cos(phi(X,Y)+l*tau(Z,T)),np.sin(beta(X,Y,Z,T))*np.sin(phi(X,Y)+l*tau(Z,T)),np.cos(beta(X,Y,Z,T))]) # Higgs isospin direction [no unit]
H=h_fun(np.sqrt(u(X,Y)**2+(v(Z,T)-R)**2))*Hhat # Full Higgs field [vev]
Hmag = np.sqrt(np.einsum('aijkl,aijkl->ijkl',H,H)) # Magnitude of Higgs field [vev]
dHhat=np.gradient(Hhat,spacing,axis={-4,-3,-2,-1}) # Derivative of Higgs isopin vector [vev^2]
ddHhat = np.einsum('bbadefg->adefg',np.gradient(dHhat,spacing,axis={-4,-3,-2,-1})) # Contracted dobule derivative of Higgs isospin vector [vev^3]

# Set up guess Abelian gauge field component
a=ken_fun(u(X,Y)**2+(v(Z,T)-R)**2)**2*2/(u(X,Y)**2+(v(Z,T)-R)**2)*np.array([-u(X,Y)*np.sin(phi(X,Y)),
u(X,Y)*np.cos(phi(X,Y)),
v(Z,T)*np.sin(tau(Z,T)),
-v(Z,T)*np.cos(tau(Z,T))])

# Set up guess  for full non-Abelian gauge field
# where the first term is the non-Abelian isospin DoF
# and the second term is the Abelian Dof.
A=-ken_fun(u(X,Y)**2+(v(Z,T)-R)**2)*(np.einsum('abc,bijkl,dcijkl->daijkl',levi_cevita_tensor(3),Hhat,dHhat))+np.einsum('aijkl,mijkl->maijkl',Hhat,a)

# Compute derivatives
dA = np.einsum('mnaijkl->mnaijkl',np.gradient(A,spacing,axis={-4,-3,-2,-1}))
dHhat=np.gradient(Hhat,spacing,axis={-4,-3,-2,-1})
ddHhat = np.einsum('bbadefg->adefg',np.gradient(dHhat,spacing,axis={-4,-3,-2,-1}))

A_guess = jnp.array(A) 
u_guess = jnp.array(Hmag) 

Hhat = jnp.asarray(Hhat)
dHhat = jnp.asarray(dHhat)
ddHhat = jnp.asarray(ddHhat)

@jit
def relaxation_step_jax(u,A,lambd,spacing,alpha):
    dx2 = spacing**2
    dA_mu = jnp.asarray(jnp.gradient(A,spacing,axis=(-4,-3,-2,-1)))
    ddA_mu = jnp.asarray(jnp.gradient(dA_mu,spacing,axis=(-4,-3,-2,-1)))
    # Compute sum of Higgs neighboring points
    u_neighbors = (
        (jnp.roll(u, 1, axis=-4) + jnp.roll(u, -1, axis=-4)) 
        + (jnp.roll(u, 1, axis=-3) + jnp.roll(u, -1, axis=-3)) 
        + (jnp.roll(u, 1, axis=-2) + jnp.roll(u, -1, axis=-2)) 
        + (jnp.roll(u, 1, axis=-1) + jnp.roll(u, -1, axis=-1))
    )
    # Compute gradient of higgs magnitude
    u_grad = jnp.asarray(jnp.gradient(u,spacing,axis=(-4,-3,-2,-1)))
    # Compute derivative term
    der_term = jnp.einsum('ijkl,aijkl->aijkl',u,ddHhat) + 2*jnp.einsum('mijkl,maijkl->aijkl',u_grad,dHhat)
    linear_term = lambd*jnp.einsum('aijkl,ijkl->aijkl',Hhat,u)*vev**2
    gauge_term = (
        2*g*jnp.einsum('abc,cijkl,mbijkl,mijkl->aijkl',levi_cevita_tensor(3),Hhat,A,u_grad) 
        + u*2*g*jnp.einsum('abc,mbijkl,mcijkl->aijkl',levi_cevita_tensor(3),A,dHhat)
        + u*g*jnp.einsum('abc,mmbijkl,cijkl->aijkl',levi_cevita_tensor(3),dA_mu,Hhat) 
        + u*g**2*jnp.einsum('abc,cde,eijkl,mbijkl,mdijkl->aijkl',levi_cevita_tensor(3),levi_cevita_tensor(3),Hhat,A,A)
    )
    rhs = lambd * jnp.einsum('aijkl,ijkl->aijkl',Hhat,u**3)
    # Compute the first term in the alpha*(...) part of the equation
    u_coeff = (1/8)*(u_neighbors+dx2*(jnp.einsum('aijkl,aijkl->ijkl',Hhat,der_term+gauge_term+linear_term-rhs)))
    # Define the new Higgs field
    u_new = u + alpha * (u_coeff-u)

    ### EoM for A ###
    # Compute sum of gauge field neighboring points
    A_neighbors = (
        (jnp.roll(A, 1, axis=-4) + jnp.roll(A, -1, axis=-4)) 
        + (jnp.roll(A, 1, axis=-3) + jnp.roll(A, -1, axis=-3)) 
        + (jnp.roll(A, 1, axis=-2) + jnp.roll(A, -1, axis=-2)) 
        + (jnp.roll(A, 1, axis=-1) + jnp.roll(A, -1, axis=-1))
    )
    ### Terms below enter into g(...) in the notation of this notebook.
    gauge_term_a = (
        g*jnp.einsum('abc,mbijkl,nncijkl->maijkl',levi_cevita_tensor(3),A,dA_mu)
        + g*jnp.einsum('abc,nbijkl,mncijkl->maijkl',levi_cevita_tensor(3),A,dA_mu)
        + g**2*jnp.einsum('maijkl,nbijkl,nbijkl->maijkl',A,A,A) 
        - g**2*jnp.einsum('naijkl,nbijkl,mbijkl->maijkl',A,A,A) 
        - 2*g*jnp.einsum('abc,nbijkl,nmcijkl->maijkl',levi_cevita_tensor(3),A,dA_mu)
    )
    # Compute the mixed gradient term
    der_term_gauge = jnp.einsum('nmnaijkl->maijkl',ddA_mu)
    # Compute the right-hand side (RHS) of the equation
    rhs_gauge = (
        -g * jnp.einsum('ijkl,abc,bijkl,mcijkl->maijkl',u**2, levi_cevita_tensor(3), Hhat, dHhat) 
        - g**2 * jnp.einsum('maijkl,ijkl->maijkl',A,u**2) 
        + g**2 * jnp.einsum('ijkl,aijkl,mbijkl,bijkl->maijkl',u**2,Hhat,A,Hhat)
    )
    # Fix Lorenz gauge (generally not enforced to minimize distance to field equation solution, so commented out)
    #gauge_fixing_term = jnp.einsum('mijkl,aijkl->maijkl',jnp.einsum('maijkl,aijkl->mijkl',A,Hhat),Hhat)
    # Compute the first term in the alpha*(...) part of the equation
    A_coeff = (1/8)*(A_neighbors-dx2*gauge_term_a-dx2*der_term_gauge+dx2*rhs_gauge)
    # Flipped sign due to appearance of minus sign in front of Laplacian in EoM 
    A_new = A + alpha * (A_coeff-A)

    # Update the fields
    u_out = u.at[1:-1, 1:-1, 1:-1, 1:-1].set(u_new[1:-1, 1:-1, 1:-1, 1:-1])
    A_out = A.at[:, :, 1:-1, 1:-1, 1:-1, 1:-1].set(A_new[:, :, 1:-1, 1:-1, 1:-1, 1:-1])

    return u_out, A_out

def relaxation_loop_jax(u, A, lambd, spacing, alpha, max_iterations, tol=1e-6):
    '''
    Input:
            u, A: initial guesses for the Higgs field and gauge field
            lambd: Higgs quartic coupling
            spacing: Grid spacing
            alpha: Mixing coefficient for field updates
            max_iterations: Maximum number of iterations
            tol: Convergence tolerance (stop when max_change < tol)

    Output:
            u_final, A_final: Relaxed fields
    '''
    def cond_fun(state):
        _, _, max_change, iter_count = state
        return (max_change > tol) & (iter_count < max_iterations)

    def body_fun(state):
        u, A, max_change, iter_count = state  # No iter_idx needed
        u_new, A_new = relaxation_step_jax(u, A, lambd, spacing, alpha)

        # Compute max change for convergence tracking
        max_change_new = jnp.maximum(
            jnp.max(jnp.abs(u_new - u)),
            jnp.max(jnp.abs(A_new - A))
        )

        return u_new, A_new, max_change_new, iter_count + 1  # Increment iter_count

    # Initial state: fields, max_change (start high), and iteration count
    init_state = (u, A, jnp.inf, 0)

    # Run the loop with dynamic stopping
    u_final, A_final, max_change_final, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)

    return u_final, A_final, max_change_final

alpha = 0.5
max_iterations = 100000
tol = 1e-5  # Convergence tolerance

# Run the relaxation loop with the stopping criterion
u_final, A_final, max_change_final = relaxation_loop_jax(u_guess, A_guess, lamb, spacing, alpha, max_iterations, tol)

u_plot = np.asarray(u_final)
A_plot = np.asarray(A_final)

np.savez('/scratch/mk7976/monopoleJAX/Relaxed_new_R='+str(R)[:3]+'.npz',H=u_plot,A=A_plot)
