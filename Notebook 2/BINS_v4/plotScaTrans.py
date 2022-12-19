import numpy as np
import matplotlib.pyplot as plt

def plotScaTrans(Nx:int,Ny:int,Lx:float,Ly:float,t:float,ng:int,c:float):
    """
    plot scalar transport for BINS

    Parameters
    ----------
    Nx,Ny : int
        number of interior points in x and y direction
    Lx,Ly : float
        physical dimension in x and y
    t : float
        current time
    ng : int
        number of ghost cells
    c : float
        scalar, size [Nx+2*ng   Ny+2*ng]
    Returns
    -------
    c_xy : float
        float, concentration at a single point

    """
    llx = ng+1
    ulx = ng+Nx
    dx = Lx/(Nx-1)
    lly = ng+1
    uly = ng+Ny
    dy = Ly/(Ny-1)
    xx = np.linspace(0,Lx-dx,Nx)
    yy = np.linspace(0,Ly-dy,Ny)

    
    [y,x] = np.meshgrid(yy,xx)
    # print(np.shape(x),np.shape(y),np.shape(c[llx:ulx,lly:uly]))
    fig20, main_ax20 = plt.subplots()
    fig20.set_size_inches(8, 8)
    main_ax20.set_xlabel('x')
    main_ax20.set_ylabel('y')
    main_ax20.set_title('Scalar Transport, t = ' + str(t))
    main_ax20.set_xlim(0, Lx)
    main_ax20.set_ylim(0, Ly)
    main_ax20.set_aspect('equal')
    cb = main_ax20.pcolor(x, y, c[llx:ulx,lly:uly], cmap='jet')
    fig20.colorbar(cb, ax=main_ax20)
    
    # output for breakthrough curve
    i0 = round((ulx-llx)/2)
    j0 = round(0.1*(uly-lly))
    return c[i0-1,j0-1]

     