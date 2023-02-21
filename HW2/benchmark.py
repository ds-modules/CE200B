'''
Simple benchmarking function for the driven cavity flow

Uses data from Erturk et. al: www.cavityflow.com

  cavity_benchmark(p,U,V,Re)

Will plot comparisons of U,V along the xy cavity midlines. It also
 returns the average error along these lines.

Darren Engwirda - 2005-06.
'''
import numpy as np
import matplotlib.pyplot as plt

def benchmark(u:float,v:float,Lx:float,Ly:float,Nx:int,Ny:int,dx:float,dy:float,nu:float):
    """
    plot benchmark BINS

    Parameters
    ----------
    u : float
        x-direction velocity, size [Nx+2*ng+1   Ny+2*ng]
    v : float
        y-direction velocity, size [Nx+2*ng   Ny+2*ng+1]
    Lx,Ly : float
        physical dimension in x and y
    Nx,Ny : int
        number of interior points in x and y direction
    dx,dy: float
        grid spacing dx and dy
    nu : float
        kinematic viscosity

    Returns
    -------
    None.

    """

    #Re = u*L/nu
    Re = Lx/nu #with u = 1
    if (Re == 1000):
      Re_index = 0
    else:
      Re_index = 0 #NOTE This is not right for other runs, but leave it here so it passes through the script. The comparison plot will not be correct.


    # Benchmark U velocity profile along x = 0.5
    yt = [
         0.00,
         0.02,
         0.04,
         0.06,
         0.08,
         0.10,
         0.12,
         0.14,
         0.16,
         0.18,
         0.20,
         0.50,
         0.90,
         0.91,
         0.92,
         0.93,
         0.94,
         0.95,
         0.96,
         0.97,
         0.98,
         0.99,
         1.00,
         ]

    xt = 0.5*np.ones(len(yt));

    # Re = [1000, 2500, 5000]
    ut = [[
         0.0000,
        -0.0757,
        -0.1392,
        -0.1951,
        -0.2472,
        -0.2960,
        -0.3381,
        -0.3690,
        -0.3854,
        -0.3869,
        -0.3756,
        -0.0620,
         0.3838,
         0.3913,
         0.3993,
         0.4101,
         0.4276,
         0.4582,
         0.5102,
         0.5917,
         0.7065,
         0.8486,
         1.0000
         ],
         [
         0.0000,
        -0.1517,
        -0.2547,
        -0.3372,
        -0.3979,
        -0.4250,
        -0.4200,
        -0.3965,
        -0.3688,
        -0.3439,
        -0.3228,
        -0.0403,
         0.4141,
         0.4256,
         0.4353,
         0.4424,
         0.4470,
         0.4506,
         0.4607,
         0.4971,
         0.5624,
         0.7704,
         1.0000
         ],
         [
         0.0000,
        -0.2223,
        -0.3480,
        -0.4272,
        -0.4419,
        -0.4168,
        -0.3876,
        -0.3652,
        -0.3467,
        -0.3285,
        -0.3100,
        -0.0319,
         0.4155,
         0.4307,
         0.4452,
         0.4582,
         0.4683,
         0.4738,
         0.4739,
         0.4749,
         0.5159,
         0.6866,
         1.0000
         ]]
       
         
    # NOTE - be careful with grid staggering
    # CE200A: This is done for you here - you don't need to change anything.
    #extract the midline for x = 0.5
    #usually ng = 1;
    # u: x-direction velocity, size [Nx+2*ng+1   Ny+2*ng]
    # v: y-direction velocity, size [Nx+2*ng   Ny+2*ng+1]
    # u: x-direction velocity, size [Nx+3   Ny+2]
    # v: y-direction velocity, size [Nx+2   Ny+3]

    imid = int(Nx/2+2)
    jmid = int(Ny/2+2)
    yy = np.linspace(dy/2,Ly-dy/2,Ny) # create grid at u points along y, midline
    ui = u[imid,1:Ny+1]
    #don't include ghost points

    fig10, main_ax10 = plt.subplots()
    fig10.set_size_inches(8, 8)
    main_ax10.set_xlabel('U velocity')
    main_ax10.set_ylabel('Y')
    main_ax10.set_title('U velocity profile')
    main_ax10.plot(ut[:][Re_index],yt,'bo')
    main_ax10.plot(ui,yy,'r')
    main_ax10.legend(['Erturk et. al.','Present study'])
       
    # Benchmark V velocity along y = 0.5
    xt = [
         0.000,
         0.015,
         0.030,
         0.045,
         0.060,
         0.075,
         0.090,
         0.105,
         0.120,
         0.135,
         0.150,
         0.500,
         0.850,
         0.865,
         0.880,
         0.895,
         0.910,
         0.925,
         0.940,
         0.955,
         0.970,
         0.985,
         1.000,
         ]
     
    yt = 0.5*np.ones(len(xt));

    vt = [[
         0.0000,
         0.1019,
         0.1792,
         0.2349,
         0.2746,
         0.3041,
         0.3273,
         0.3460,
         0.3605,
         0.3705,
         0.3756,
         0.0258,
        -0.4028,
        -0.4407,
        -0.4803,
        -0.5132,
        -0.5263,
        -0.5052,
        -0.4417,
        -0.3400,
        -0.2173,
        -0.0973,
         0.0000
        ],
        [
         0.0000,
         0.1607,
         0.2633,
         0.3238,
         0.3649,
         0.3950,
         0.4142,
         0.4217,
         0.4187,
         0.4078,
         0.3918,
         0.0160,
        -0.3671,
        -0.3843,
        -0.4042,
        -0.4321,
        -0.4741,
        -0.5268,
        -0.5603,
        -0.5192,
        -0.3725,
        -0.1675,
         0.0000
         ],
         [
         0.0000,
         0.2160,
         0.3263,
         0.3868,
         0.4258,
         0.4426,
         0.4403,
         0.4260,
         0.4070,
         0.3878,
         0.3699,
         0.0117,
        -0.3624,
        -0.3806,
        -0.3982,
        -0.4147,
        -0.4318,
        -0.4595,
        -0.5139,
        -0.5700,
        -0.5019,
        -0.2441,
         0.0000
         ]]
         
         
    # NOTE - be careful with grid staggering
    # CE200A: This is done for you here - you don't need to change anything.
    #extract the midline for y = 0.5
        
    vi = v[1:Nx+1,jmid]
    xx = np.linspace(dx/2,Lx-dx/2,Nx) # create grid at cell centers

    fig11, main_ax11 = plt.subplots()
    fig11.set_size_inches(8, 8)
    main_ax11.set_xlabel('V velocity')
    main_ax11.set_ylabel('X')
    main_ax11.set_title('V velocity profile')
    main_ax11.plot(xt,vt[:][Re_index],'bo')
    main_ax11.plot(xx,vi,'r')
    main_ax11.legend(['Erturk et. al.','Present study'])
