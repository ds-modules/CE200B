import numpy as np 
from utils import np_arange

def IC(Lx:float,Ly:float,Nx:int,Ny:int,ng:int,IC_choice:int,T:float,dt:float):
    """
    Initialize u,v,p,temp and c
    The base size of variables is N+2*ng

    Each dimension looks like
    [guardcells domain guardcells]
    
    Parameters
    -----------
    Lx,Ly: length of the domain
    Nx,Ny: number of grid cells
    ng: number of guard cells
    IC_choice: integer to choose among initial conditions defined here    

    inc: Time increment for plotting

    Returns
    ---------
    u: x-direction velocity, size [Nx+2*ng+1   Ny+2*ng]
    v: y-direction velocity, size [Nx+2*ng   Ny+2*ng+1]
    p: Pressure, size [Nx+2*ng   Ny+2*ng]
    temp: Temperature, size [Nx+2*ng   Ny+2*ng]
    """

    # Initialize the returns
    u = np.zeros((Nx+2*ng+1,Ny+2*ng))
    v = np.zeros((Nx+2*ng,Ny+2*ng+1))
    p = np.zeros((Nx+2*ng,Ny+2*ng))
    c = np.zeros((Nx+2*ng,Ny+2*ng))
    temp = np.zeros((Nx+2*ng,Ny+2*ng))
    temp_null = 273

    # Flag to turn gravity on/off
    aa = 0

    # Time increment for plotting
    inc = 1
    
    # Other variables
    dx = Lx/Nx
    dy = Ly/Ny
    li = ng+1
    uix = ng+Nx
    uiy = ng+Ny
    
    # Simple if-else instead of switch
    if IC_choice == 1:
        c[int(4/64*Nx):int(8/64*Nx),int(56/64*Ny):int(60/64*Ny)] = 1
        inc = 100
        
    elif IC_choice == 2:
        # Velocity is all ones
        c[int(9/64*Nx):int(20/64*Nx),int(9/64*Ny):int(20/64*Ny)] = 1
        u = u+1
        v = v+1

    elif IC_choice == 3: #NOTE: check if this works for different Nx, Ny
        # Taylor-Green vortex
        N = min(Nx,Ny)
        c[:,int(Nx/3)-1:int(Nx/3)+1] = 1#np.eye(N+2*ng)
        # u
        # xx = np_arange(0,Lx,Lx/Nx)
        # yy = np_arange(dy/2,Ly-dy/2,(Ly-dy)/(Ny-1))
        xx = np.linspace(0,Lx,Nx+1)
        yy = np.linspace(dy/2,Ly-dy/2,Ny)
        y,x = np.meshgrid(yy,xx)
        u[li-1:uix+1,li-1:uiy] = np.multiply(np.sin(x),np.cos(y))
        # v
        # xx = np_arange(dx/2,Lx-dx/2,(Lx-dx)/(Nx-1))
        # yy = np_arange(0,Ly,Ly/Ny)
        xx = np.linspace(dx/2,Lx-dx/2,Nx);
        yy = np.linspace(0,Ly,Ny+1);
        y,x = np.meshgrid(yy,xx)
        v[li-1:uix,li-1:uiy+1] = np.multiply(-np.cos(x),np.sin(y))
        # p 
        # xx = np_arange(dx/2,Lx-dx/2,(Lx-dx)/(Nx-1))
        # yy = np_arange(dy/2,Ly-dy/2,(Ly-dy)/(Ny-1))
        xx = np.linspace(dx/2,Lx-dx/2,Nx);
        yy = np.linspace(dy/2,Ly-dy/2,Ny);
        y,x = np.meshgrid(yy,xx)
        p[li-1:uix,li-1:uiy] = 0.25*(np.cos(2*x)  + np.cos(2*y))
        inc = 20
        
    
        
    elif IC_choice == 4: 
        # Shear flow (Original Maltab code from Dr. Colella at LBNL)
        c[19:34,:] = 1
        # u
        xx = np_arange(0,Lx,Lx/Nx)
        yy = np_arange(dy/2,Ly-dy/2,(Ly-dy)/(Ny-1))
        y,x = np.meshgrid(yy,xx)
        u[li-1:uix+1,li-1:li+int(Ny/2)-1] = np.tanh(30*(y[:,0:int(Ny/2)] - (1/4)))
        u[li-1:uix+1,li+int(Ny/2)-1:uiy] = np.tanh(30*((3/4)-y[:,int(Ny/2):Ny]))
        # v
        xx = np_arange(dx/2,Lx-dx/2,(Lx-dx)/(Nx-1))
        yy = np_arange(0,Ly,Ly/Ny)
        y,x = np.meshgrid(yy,xx)
        v[li-1:uix,li-1:uiy+1] = (1/20)*np.sin(2*np.pi*x)      
        p = np.zeros(np.shape(p)) 
        inc = 10
        
    elif IC_choice == 5:
        # Infinite channel
        u = u+1
        c[li+6:li+11,li+1:uiy-2] = 1
        inc = 20
        
    elif IC_choice == 6:
        # Density/temperature instability
        aa = 1
        temp[:,:] = temp_null
        temp[:,0:int(Ny/2)] = 293
        temp[int(Nx/2)-4:int(Nx/2)+3,int(Ny/2)-2:int(Ny/2)] = temp_null
        c[:,int(Ny/2):] = 1
        c[int(Nx/2)-4:int(Nx/2)+3,int(Ny/2)-2:int(Ny/2)] = 1
        inc = 35
        
    elif IC_choice == 7:
        # Couette flow
        c[int(4/64*Nx):int(7/64*Nx),int(4/64*Ny):int(7/64*Ny)] = 1
        inc = 50

        
    # Return
    return u,v,p,c,temp,temp_null,aa,inc
