import numpy as np
import matplotlib.pyplot as plt
from utils import np_arange
from benchmark import benchmark

def plotSoln(u:float,v:float,ng:int,Lx:float,Ly:float,Nx:int,Ny:int,dx:float,dy:float,
             nu:float,t:float,T:float,IC_choice:int):
    """
    plot solution to BINS

    Parameters
    ----------
    u : float
        x-direction velocity, size [N+2*ng+1   N+2*ng]
    v : float
        y-direction velocity, size [N+2*ng   N+2*ng+1]
    ng : int
        number of ghost cells
    Lx,Ly : float
        physical dimension in x and y
    Nx,Ny : int
        number of interior points in x and y direction
    dx,dy : float
        grid spacing dx and dy
    nu : float
        kinematic viscosity
    IC_choice : int
        choice of initial condition

    Returns
    -------
    None.

    """

    # limits of internal (non-ghostcell) data
    llx = ng + 1
    lly = ng + 1
    ulx = ng + Nx
    uly = ng + Ny
    
  #  # u-velocity
  #  # setup the grid
  #  # xx = np_arange(0, Lx, Lx/Nx)
  #  # yy = np_arange(dy/2, Ly-dy/2, (Ly-dy)/(Ny-1))
  #  xx = np.linspace(0,Lx,Nx+1) # grid of x faces
  #  yy = np.linspace(dy/2,Ly-dy/2,Ny)
  #  y,x = np.meshgrid(yy,xx)
    
    # plot
  #  fig1, main_ax1 = plt.subplots()
  #  fig1.set_size_inches(8, 8)
  #  main_ax1.set_xlabel('x')
  #  main_ax1.set_ylabel('y')
  #  main_ax1.set_title('u velocity, t = ' + str(t))
  #  main_ax1.set_xlim(0, Lx)
  #  main_ax1.set_ylim(0, Ly)
  #  c = main_ax1.pcolor(x, y, u[llx-1:ulx+1,lly-1:uly], cmap='RdBu')
  #  fig1.colorbar(c, ax=main_ax1)
  #  main_ax1.set_aspect('equal')

  #  # v-velocity
  #  # setup the grid
  #  # xx = np_arange(dx/2, Lx-dx/2, (Lx-dx)/(Nx-1))
  #  # yy = np_arange(0, Ly, Ly/Ny)
  #  xx = np.linspace(dx/2,Lx-dx/2,Nx); # grid of y faces
  #  yy = np.linspace(0,Ly,Ny+1);
  #  y,x = np.meshgrid(yy,xx)
    
    # plot
  #  fig2, main_ax2 = plt.subplots()
  #  fig2.set_size_inches(8, 8)
  #  main_ax2.set_xlabel('x')
  #  main_ax2.set_ylabel('y')
  #  main_ax2.set_title('v velocity, t = ' + str(t))
  #  main_ax2.set_xlim(0, Lx)
  #  main_ax2.set_ylim(0, Ly)
  #  c = main_ax2.pcolor(x, y, v[llx-1:ulx,lly-1:uly+1], cmap='RdBu')
  #  fig2.colorbar(c, ax=main_ax2)
  #  main_ax2.set_aspect('equal')

   
    # speed and streamlines
    # average velocities to cell centers
    endi = np.shape(u)[0]
    endj = np.shape(v)[1]
    midU = 0.5*(u[0:endi-1,:] + u[1:endi,:])
    midV = 0.5*(v[:,0:endj-1] + v[:,1:endj])
    
    # setup the grid
    # xx = np_arange(dx/2, Lx-dx/2, (Lx-dx)/(Nx-1))
    # yy = np_arange(dy/2, Ly-dy/2, (Ly-dy)/(Ny-1))
    xx = np.linspace(dx/2,Lx-dx/2,Nx); # grid of cell centers
    yy = np.linspace(dy/2,Ly-dy/2,Ny);
    y,x = np.meshgrid(yy,xx)
    
    # plot contours of speed
    speed = np.sqrt(np.multiply(midU,midU)+np.multiply(midV,midV))
    fig3, main_ax3 = plt.subplots()
    fig3.set_size_inches(8, 8)
    main_ax3.set_xlabel('x')
    main_ax3.set_ylabel('y')
    main_ax3.set_xlim(0, Lx)
    main_ax3.set_ylim(0, Ly)
    main_ax3.set_title('speed with streamlines, t = ' + str(t))
    c = main_ax3.contourf(xx, yy, np.transpose(speed[llx-1:ulx,lly-1:uly]), cmap='GnBu')
    fig3.colorbar(c, ax=main_ax3)
    main_ax3.set_aspect('equal')

    # setup streamlines
    # sx = np_arange(dx/2, Lx-dx/2, (Lx-dx)/((Nx/4)-1))
    # sy = np_arange(dy/2, Ly-dy/2, (Ly-dy)/((Ny/4)-1))
    sx = np.linspace(dx/2,Lx-dx/2,round(Nx/4)); # where streamlines start
    sy = np.linspace(dy/2,Ly-dy/2,round(Ny/4));
    sxx,syy = np.meshgrid(sx,sy)
    
    # plot streamlines
    main_ax3.streamplot(np.transpose(x), np.transpose(y),
                        np.transpose(midU[llx-1:ulx,lly-1:uly]), 
                        np.transpose(midV[llx-1:ulx,lly-1:uly]))
    
    # Vorticity
  #  endj = np.shape(u)[1]
  #  endi = np.shape(v)[0]
  #  # du/dy at lower left cell corners
  #  dudy = (u[:,1:endj]-u[:,0:endj-1])/dy
  #  # dv/dx at lower left cell corners
  #  dvdx = (v[1:endi,:]-v[0:endi-1,:])/dx
  #  # just the inner (non-ghost) points
  #  vorticity = dvdx[llx-2:ulx-1,lly-1:uly] - dudy[llx-1:ulx,lly-2:uly-1]
    
  #  # setup the grid
  #  xx = np_arange(0,Lx-dx,(Lx-dx)/(Nx-1))
  #  yy = np_arange(0,Ly-dy,(Ly-dy)/(Ny-1))
  #  y,x = np.meshgrid(yy,xx)
    
    # plot
  #  fig4, main_ax4 = plt.subplots()
  #  fig4.set_size_inches(8, 8)
  #  main_ax4.set_xlabel('x')
  #  main_ax4.set_ylabel('y')
  #  main_ax4.set_title('vorticity, t = ' + str(t))
  #  main_ax4.set_xlim(0, Lx)
  #  main_ax4.set_ylim(0, Ly)
  #  c = main_ax4.pcolor(x, y, vorticity, cmap='RdBu')
  #  fig4.colorbar(c, ax=main_ax4)
  #  main_ax4.set_aspect('equal')

    if (IC_choice == 1 and (t > T)): #lid-driven cavity case
        benchmark(u,v,Lx,Ly,Nx,Ny,dx,dy,nu)