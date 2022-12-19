import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
    
#def poissonSolve(rhs = None,h = None,BC = None,ng = None,N = None,LaP = None):
def poissonSolve(rhs:float,dx:float,dy:float,BC:complex,ng:int,Nx:int,Ny:int,A:float):
    """
    solve the Poisson equation:   d^2 solution / dx^2 + d^2 solution/dy^2 = rhs
    Numerically, this equation results in a matrix inversion:
    A_ij solution_j = rhs_i
    for more details, see Section 3.4 on page 8

    Parameters
    ----------
    rhs : float
        right hand side to Poisson equation, size [N+2*ng   N+2*ng]
    dx,dy : float
        grid size
    BC : complex
        array defining boundary conditions, size [4]
    ng : int
        number of guardcells
    Nx,Ny : int
        number of points in the grid
    A : float
        matrix defining Poisson operator

    Returns
    -------
    solution : float
        the solution to the Poisson equation, the new pressure, size [Nx+2*ng   Ny+2*ng]

    """
    
    # Initialize the solution vector
    solution = np.copy(rhs)

    # make the vector f that defines our right hand side of the equation
    f = np.reshape(rhs[ng:ng+Nx,ng:ng+Ny], (Nx*Ny,1), order='F') #TINA check this

    # Use solver from scipy
    # solnVector = spsolve(-A, f)
    solnVector, istop, itn, r1norm = lsqr(-A, f,atol=1e-16,btol=1e-16,iter_lim=100000.0)[:4]

    #TINA check this
    # convert back from array to size
    solution[ng:ng+Nx,ng:ng+Ny] = np.reshape(solnVector,(Nx,Ny),order='F')

    return solution
