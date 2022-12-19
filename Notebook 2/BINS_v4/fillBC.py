import numpy as np 

def fillBC(u:float,v:float,p:float,c:float,temp:float,ng:int,Nx:int,Ny:int,BC:float,IBM:int):
    """
    Fills the ghost cells of u, v, and p to enforce boundary conditions

    Parameters
    -----------
    u: x-direction velocity, size [Nx+2*ng+1   Ny+2*ng]
    v: y-direction velocity, size [Nx+2*ng   Ny+2*ng+1]
    p: Pressure, size [Nx+2*ng   Ny+2*ng]
    ng: number of ghost cells
    Nx,Ny: number of interior points in x and y direction
    BC: boundary condition tags in x and y direction, size [4]    
    IBM: flag to turn on IBM - immersed obstacle

    Returns
    ---------
    uOut: x-direction velocity with updated ghost cells, size [Nx+2*ng+1   Ny+2*ng]
    vOut: y-direction velocity with updated ghost cells, size [Nx+2*ng   Ny+2*ng+1]
    pOut: pressure with updated ghost cells, size [Nx+2*ng   Ny+2*ng]
    cOut: scalar with updated ghost cells, size [Nx+2*ng   Ny+2*ng]
    tOut: temperature with updated ghost cells, size [Nx+2*ng   Ny+2*ng]
    """
    
    # Initialize the variables
    uOut = np.copy(u)
    vOut = np.copy(v)
    pOut = np.copy(p)
    cOut = np.copy(c)
    tOut = np.copy(temp)
    
    # Loop over the ghost cells
    for gcell in range(0,ng):
        
        # Lower x-BC
        if (BC[0,0].real == 0) and (BC[0,0].imag == 1):
            # Periodic, filled with upper x inner data
            uOut[gcell,:] = u[Nx+gcell,:]
            vOut[gcell,:] = v[Nx+gcell,:] 
            pOut[gcell,:] = p[Nx+gcell,:]
            cOut[gcell,:] = c[Nx+gcell,:]
            tOut[gcell,:] = temp[Nx+gcell,:]
        else:
            # Wall moving at speed BC parallel to itself
            uOut[gcell,:] = 0
            uOut[gcell+1,:] = 0
            vOut[gcell,:] = 2*BC[0,0].real - v[ng,:]
            pOut[gcell,:] = p[ng,:] 
            cOut[gcell,:] = c[ng,:]
            tOut[gcell,:] = temp[ng,:]
        # Upper x-BC
        if (BC[0,1].real == 0) and (BC[0,1].imag == 1):
            # Periodic, filled with lower x inner data
            uOut[Nx+ng+gcell,:] = u[ng+gcell,:]
            uOut[Nx+ng+gcell+1,:] = u[ng+gcell+1,:]
            vOut[Nx+ng+gcell,:] = v[ng+gcell,:]
            pOut[Nx+ng+gcell,:] = p[ng+gcell,:]
            cOut[Nx+ng+gcell,:] = c[ng+gcell,:]
            tOut[Nx+ng+gcell,:] = temp[ng+gcell,:]
        else:
            # Wall moving at speed BC parallel to itself
            uOut[Nx+ng+gcell,:] = 0
            uOut[Nx+ng-1,:] = 0
            vOut[Nx+ng+gcell,:] = 2*BC[0,1].real - v[ng+Nx-1,:]
            pOut[Nx+ng+gcell,:] = p[ng+Nx-1,:]
            cOut[Nx+ng+gcell,:] = c[ng+Nx-1,:]
            tOut[Nx+ng+gcell,:] = temp[ng+Nx-1,:]
            
        # Lower y-BC
        if (BC[0,2].real == 0) and (BC[0,2].imag == 1):
            # Periodic, filled with lower x inner data
            uOut[:,gcell] = u[:,Ny+gcell]
            vOut[:,gcell] = v[:,Ny+gcell]
            pOut[:,gcell] = p[:,Ny+gcell]
            pOut[gcell,gcell] = pOut[gcell+Nx,gcell+Ny]
            cOut[:,gcell] = c[:,Ny+gcell]
            cOut[gcell,gcell] = cOut[gcell+Nx,gcell+Ny]
            tOut[:,gcell] = temp[:,Ny+gcell]
            tOut[gcell,gcell] = temp[gcell+Nx,gcell+Ny]
        else:
            # Wall moving at speed BC parallel to itself
            uOut[:,gcell] = 2*BC[0,2].real - u[:,ng]           
            vOut[:,gcell] = 0
            vOut[:,gcell+1] = 0
            pOut[:,gcell] = p[:,ng]
            pOut[gcell,gcell] = p[ng,ng]
            cOut[:,gcell] = c[:,ng]
            cOut[gcell,gcell] = c[ng,ng]
            tOut[:,gcell] = temp[:,ng]
            tOut[gcell,gcell] = temp[ng,ng]
            
        # Upper y-BC
        if (BC[0,3].real == 0) and (BC[0,3].imag == 1):
            # Periodic, filled with lower x inner data
            uOut[:,Ny+ng+gcell] = u[:,ng+gcell]
            vOut[:,Ny+ng+gcell] = v[:,ng+gcell]
            vOut[:,Ny+ng+gcell+1] = v[:,ng+gcell+1]
            pOut[:,Ny+ng+gcell] = p[:,ng+gcell]
            cOut[:,Ny+ng+gcell] = c[:,ng+gcell]
            tOut[:,Ny+ng+gcell] = temp[:,ng+gcell]
        else:
            # Wall moving at speed BC parallel to itself
            uOut[:,Ny+ng+gcell] = 2*BC[0,3].real - u[:,Ny+gcell]
            vOut[:,Ny+ng+gcell] = 0
            vOut[:,Ny+ng-1] = 0
            pOut[:,Ny+ng+gcell] = p[:,Ny+gcell] 
            cOut[:,Ny+ng+gcell] = c[:,Ny+gcell]
            tOut[:,Ny+ng+gcell] = temp[:,Ny+gcell]

    # Set IBM boundary conditions for immersed obstacle
    if IBM == 1: 
	#note - this will not be precise because of the staggering
        uOut[int(Nx/4):2*int(Nx/5),0:2*int(Ny/5)] = 0 #set as no slip
        vOut[int(Nx/4):2*int(Nx/5),0:2*int(Ny/5)] = 0 #set as no slip
        cOut[int(Nx/4):2*int(Nx/5),0:2*int(Ny/5)] = 0 #should be set as zero gradient
        tOut[int(Nx/4):2*int(Nx/5),0:2*int(Ny/5)] = 0
        # pOut[int(Nx/4):int(Nx/2),int(Ny/4)] = 0 #should be set as zero gradient

    return uOut,vOut,pOut,cOut,tOut
