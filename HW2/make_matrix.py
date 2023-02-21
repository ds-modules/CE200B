from scipy.sparse import spdiags
from scipy import sparse
import numpy as np 

def LaP1D(N:int,h:float,BC:complex): #TINA pass in either Nx,Ny, dx,dy
    """
    Form a matrix defining the 1D Laplace operator, with BCs

    Parameters
    ----------
    N : int
        number of rows
    h : float
        grid spacing 
    BC : complex
        boundary condition for this dimension

    Returns
    -------
    A : float
        matrix defining the 1D Laplace operator, with BCs, 
        for this dimension,size [N N] #TINA
    """
    
    if BC.real==0 and BC.imag==1:
        
        # Dirichlet BC for pressure, p = (periodic p)
        BCval = 2
        
        # Create the sparse matrix
        row1 = [-1,BCval,0]
        rown = [0,BCval,-1]
        data = np.ones((N-2,1))*[-1,2,-1]
        data = np.vstack([row1,data])
        data = np.vstack([data,rown])
        diags = np.array([-1,0,1])
        A = spdiags(np.transpose(data),diags,N,N)/(h*h)
        
        # Find the periodic point [this is not the same as pinning]
        Amat = A.tolil()
        Amat[0,N-1] = -1/(h*h) #this is needed to create the proper periodic BCs matrix
        Amat[N-1,0] = -1/(h*h)
        
    else:
        
        # Neumann BC for pressure, dpdn = 0
        BCval = 1
        
        # Create the sparse matrix
        row1 = [-1,BCval,0]
        rown = [0,BCval,-1]
        data = np.ones((N-2,1))*[-1,2,-1]
        data = np.vstack([row1,data])
        data = np.vstack([data,rown])
        diags = np.array([-1,0,1])
        A = spdiags(np.transpose(data),diags,N,N)/(h*h)
        
        # Find the periodic point
        Amat = A.tolil()
    
    return Amat


def is_pos_def(A):
    """
    Checks if a matrix is symmetric and positive definite

    Parameters
    ----------
    A : float
        Matrix of interest

    Returns
    -------
    bool
        Returns true or false

    """
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def make_matrix(Nx:int,Ny:int,dx:float,dy:float,BC:complex): #TINA added Nx,Ny,dx,dy
    """
    Form the matrix inversion problem that solves for the pressure
	Laplacian(p) = f

    Parameters
    ----------
    Nx : int
        number of grid points in x direction
    Ny : int
        number of grid points in x direction
    dx : float
        grid spacing dx 
    dy : float
        grid spacing dy 
    BC : complex
        array of boundary conditions

    Returns
    -------
    poisSolve : TYPE
        A structure containing either
        if the BCs make the Laplacian operator positive definite 
        (true for the wall BCs)
        perp: a sparse matrix defining how the indices of Rp have been rearranged
        Rp: a sparse matrix defining the Laplace operator with rearranged indices

        or, if the BCs do not,
        (true for the periodic BCs)
        A: a sparse matrix defining the Laplace operator


    """

    bigM = Nx*Ny #define total number of elements in each row/column of the big matrix

    # Construct the matrix A in sparse form       
    # Note: the first .kron creates a tridiagonal in blocks of Nx rows
    # the second .kron creates a main diagonal and 2 off diagonals that are all the same, need to pass dy into this
    # Note - A accounts for the pressure BCs - so the lower x and lower y boundary condition flags are used to decide whether to
    # set periodic or dp/dn = 0 conditions for the pressure BCs
    A = sparse.kron(sparse.csr_matrix(np.eye(Ny)),LaP1D(Nx,dx,BC[0,0])) + \
        sparse.kron(LaP1D(Ny,dy,BC[0,2]),sparse.csr_matrix(np.eye(Nx)))


    # Currently not using this because a least-squares matrix solution is implemented              
    # Check for symmetric and positive definite
    # if not is_pos_def(A):
    #     # Pin corner so matrix is solvable
    #     A[0,:] = 0
    #     A[0,0] = -1/(dx*dx)
    
    # Return variables
    return A
