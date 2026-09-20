import hoomd
import gsd.hoomd
import math
import numpy as np
import io
import os
import sys
import time
import glob
import os.path as path
import matplotlib.pyplot as plt
import h5py
import warnings
import fresnel
import IPython
import packaging.version
import PIL

#----------------------------------------------------------------------------
barh0scale = 1.4 # L0,Bar
Yh0scale = 1.0 # L0,Y
folderprefix = f'./output/'
os.makedirs(folderprefix, exist_ok=True)

# magnet profile
halfl0 = 1 # unit branch length L0 in cm
thickwall = 0.08 # magnet holder wallthickness in cm
magr = 1/16*2.54 + thickwall # branch-tip radius (magnet radius + magnet holder wallthickness) in cm
tm = 0.25*25.4   # magnet height in mm 

# standard units
barl0_m = halfl0*2/100 #m
halfl0_m = barl0_m/2 #m
magr_m = magr/100 #m

Br = 1.31 # in T (~N42)
mu0 = 4*np.pi*1e-7    # permeability of vacuum (H/m), H = J/A2
M = Br/mu0    # magnetization (A/m)
V = (tm/1000)*np.pi*((magr-thickwall)/100)**2    # bar magnet volume (m3)
mag_amp = M*V  # magnet moment (A*m2)
mag_mass = 0.377/1000  # in kg (from K&J magnet spec.s)
# mag_J = 0.5*mag_mass*((magr-thickwall)/100)**2 # in kg*m^2

#---------------------------------
resizeLx = 1.0
resizeLy = 3.0 # vacuum, lattice, vacuum along y-dir
mdfile = f"MD_npt0_eq4.gsd"

simrandseed = np.random.randint(65535)
print(f'simrandseed: {simrandseed}.')
r_mag_cutoff = 7.0/100 # m
thisdt = 2e-7 # s/setp
#----------------------------------------------------------------------------
xyratio = 1.0

device = fresnel.Device()
tracer = fresnel.tracer.Path(device=device, w=int(800*xyratio), h=800)

FRESNEL_MIN_VERSION = packaging.version.parse("0.13.0")
FRESNEL_MAX_VERSION = packaging.version.parse("0.14.0")

def render_m2cm(snapshot, savefig='out.png', Lx=None, Ly=None, exist_bar=False, exist_Y=False, exist_cross=False):
    """ replace exist_bar, exist_Y, exist_cross with (typeid + 1)
    """ 
    if ('version' not in dir(fresnel) or packaging.version.parse(
            fresnel.version.version) < FRESNEL_MIN_VERSION
            or packaging.version.parse(
                fresnel.version.version) >= FRESNEL_MAX_VERSION):
        warnings.warn(
            f"Unsupported fresnel version {fresnel.version.version} - expect errors."
        )

    if (not Lx) and (not Ly):
        Lx = snapshot.configuration.box[0]*100
        Ly = snapshot.configuration.box[1]*100

    scene = fresnel.Scene(device) 

    # bar
    if exist_bar:
        vertices= [(-halfl0*barh0scale,0), (halfl0*barh0scale,0)]
        barid = exist_bar -1
        centerind = np.where(snapshot.particles.typeid==barid)[0]
        Ncenter = len(centerind)
        geometry = fresnel.geometry.Polygon(scene, vertices, N=Ncenter, rounding_radius=magr)
        geometry.material = fresnel.material.Material(color=fresnel.color.linear([0.5,0.5,0.5]), roughness=0.5)
        geometry.position[:] = snapshot.particles.position[centerind,:-1]*100
        geometry.angle[:] = 2*np.arctan2(snapshot.particles.orientation[centerind,-1], snapshot.particles.orientation[centerind,0])
        geometry.outline_width = 0.0

    # cross (dummy)
    if exist_cross:
        vertices= [(-halfl0,0), (halfl0,0)]
        crossid = exist_cross -1
        centerind = np.where(snapshot.particles.typeid==crossid)[0]
        Ncenter = len(centerind)
        geometry2 = fresnel.geometry.Polygon(scene, vertices, N=2*Ncenter, rounding_radius=magr)
        geometry2.material = fresnel.material.Material(color=fresnel.color.linear([0.631, 0.404, 0.247]), roughness=0.5)
        geometry2.position[:] = (snapshot.particles.position[centerind,:-1]).repeat(2,axis=0)*100
        # h bar
        aa = 2*np.arctan2(snapshot.particles.orientation[centerind,-1], snapshot.particles.orientation[centerind,0])
        # v bar
        bb = aa+np.pi/2
        # combine angle together
        geometry2.angle[:] = np.vstack([aa,bb]).T.flatten()
        geometry2.outline_width = 0.0

    # Y
    if exist_Y:
        vertices= [(0,0), (halfl0*Yh0scale,0)]
        Yid = exist_Y -1
        centerind = np.where(snapshot.particles.typeid==Yid)[0]
        Ncenter = len(centerind)
        geometry3 = fresnel.geometry.Polygon(scene, vertices, N=3*Ncenter, rounding_radius=magr)
        geometry3.material = fresnel.material.Material(color=fresnel.color.linear([0.25,0.5,0.9]), roughness=0.5)
        geometry3.position[:] = (snapshot.particles.position[centerind,:-1]).repeat(3,axis=0)*100
        # h bar
        aa = 2*np.arctan2(snapshot.particles.orientation[centerind,-1], snapshot.particles.orientation[centerind,0])
        # v2 bar
        bb = aa-2*np.pi/3
        # v3 bar
        cc = aa+2*np.pi/3
        # combine angle together (angle is the rotation angle of the bar about [0,0])
        geometry3.angle[:] = np.vstack([aa,bb,cc]).T.flatten()
        geometry3.outline_width = 0.0


    box = fresnel.geometry.Box(scene,
                            (Lx, Ly, 0.0, 0.0, 0.0, 0.0),
                            box_radius=.02)

    scene.lights = [
        fresnel.light.Light(direction=(0, 0, 1),
                            color=(0.8, 0.8, 0.8),
                            theta=math.pi),
        fresnel.light.Light(direction=(1, 1, 1),
                            color=(1.1, 1.1, 1.1),
                            theta=math.pi / 3)
    ]
    scene.camera = fresnel.camera.Orthographic(position=(0, 0, Lx * 1.1),
                                            look_at=(0, 0, 0),
                                            up=(0, 1, 0),
                                            height=Ly * 1.1)
    scene.background_alpha = 1
    scene.background_color = (1, 1, 1)
    image = tracer.sample(scene, samples=500)._repr_png_()
    if savefig:
        with open(savefig, "wb") as png:
            png.write(image)
    return IPython.display.Image(image)


#------------- bar unit element info ------------
mass_per_particle_bar = 2*mag_mass #[need to change for multiple branches]
# create rigid body
constituent_mass_bar = [mag_mass, mag_mass]
constituent_poss_bar = [(-halfl0_m*barh0scale, 0, 0), (halfl0_m*barh0scale, 0, 0)]
constituent_orient_bar = [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]
# Compute the moment of inertia of the rigid particle
I_bar = np.zeros(shape=(3, 3))
for i in range(len(constituent_poss_bar)):
    mass = constituent_mass_bar[i]
    r = constituent_poss_bar[i]
    I_bar += mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))

#------------- cross unit element info ------------
mass_per_particle_cross = 4*mag_mass #[need to change for multiple branches]
# create rigid body
constituent_mass_cross = [mag_mass, mag_mass, mag_mass, mag_mass]
constituent_poss_cross = [(-halfl0_m, 0, 0), (halfl0_m, 0, 0), (0, -halfl0_m, 0), (0, halfl0_m, 0)]
constituent_orient_cross = [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 
                    (np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4)), (np.cos(np.pi/4), 0.0, 0.0, np.sin(np.pi/4))]
# Compute the moment of inertia of the rigid particle
I_cross = np.zeros(shape=(3, 3))
for i in range(len(constituent_poss_cross)):
    mass = constituent_mass_cross[i]
    r = constituent_poss_cross[i]
    I_cross += mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))

#------------- Y unit element info ------------
mass_per_particle_Y = 3*mag_mass #[need to change for multiple branches]
# create rigid body
constituent_mass_Y = [mag_mass, mag_mass, mag_mass]
constituent_poss_Y = [(halfl0_m*Yh0scale, 0, 0), 
                    (-halfl0_m*np.cos(np.pi/3)*Yh0scale, -halfl0_m*np.sin(np.pi/3)*Yh0scale, 0), 
                    (-halfl0_m*np.cos(np.pi/3)*Yh0scale, halfl0_m*np.sin(np.pi/3)*Yh0scale, 0)]
constituent_orient_Y = [(1.0, 0.0, 0.0, 0.0), (np.cos(-np.pi/3), 0.0, 0.0, np.sin(-np.pi/3)), (np.cos(np.pi/3), 0.0, 0.0, np.sin(np.pi/3))]
# Compute the moment of inertia of the rigid particle
I_Y = np.zeros(shape=(3, 3))
for i in range(len(constituent_poss_Y)):
    mass = constituent_mass_Y[i]
    r = constituent_poss_Y[i]
    I_Y += mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))

#---------------------------------------------
# id for plotting
plot_barid = 0
plot_Yid = 3
plot_crossid = 7

#-----------------------------------------------
gpu = hoomd.device.CPU()

# id in sim
barid = 0
Yid = 3
crossid = 7

N_particles_bar = 396 + 11
N_particles_Y = 264
N_particles_cross = 0

simMD = hoomd.Simulation(device=gpu, seed=999)
communicator = simMD.device.communicator
rank = communicator.rank
simMD.create_state_from_gsd(filename=mdfile)
# this line must appear before if rank==0:,
# otherwise, rank 0 is waiting for snapshot info from other ranks forever
curr_snapshot = simMD.state.get_snapshot()
Lx = simMD.state.box.Lx
Ly = simMD.state.box.Ly
ini_positions = curr_snapshot.particles.position
ini_quaternions = curr_snapshot.particles.orientation

#------------------- create a MD.gsd from input MD.gsd -----
frame = gsd.hoomd.Frame()
frame.particles.types = ['Bar', 'M1_bar', 'M2_bar'] + ['Y', 'M1_Y', 'M2_Y', 'M3_Y'] + ['Cross', 'M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'] #[add]
bar_ids = np.where(curr_snapshot.particles.typeid==barid)[0]
Y_ids = np.where(curr_snapshot.particles.typeid==Yid)[0]
cross_ids = np.where(curr_snapshot.particles.typeid==crossid)[0]

# Adding vacuum below the lattice removes the confinement of the boundary Bar units.
# Copy the top-boundary Bar units to the bottom boundary to restore confinement.
add_ind = np.argsort(ini_positions[bar_ids, 1])[::-1][:11]
add_quater = ini_quaternions[bar_ids][add_ind]
add_pos = ini_positions[bar_ids][add_ind] - np.array([0, Ly, 0])
suffix = '_vacuum'

N_particles = N_particles_bar + N_particles_Y + N_particles_cross
frame.particles.N = N_particles
frame.particles.position = np.vstack([ini_positions[bar_ids], add_pos, ini_positions[np.hstack([Y_ids, cross_ids])]])
frame.particles.velocity = np.zeros([N_particles, 3])
frame.particles.typeid = [0] * N_particles_bar + [3] * N_particles_Y + [7] * N_particles_cross # [check] will be saved as 0 0 ... 0 3 3 ... 3 1 2 1 2 ... 1 2 4 5 6 7 4 5 6 7 ... 

frame.configuration.box = [Lx*resizeLx, Ly*resizeLy, 0, 0, 0, 0]

# mass
frame.particles.mass = [mass_per_particle_bar]*N_particles_bar + [mass_per_particle_Y]*N_particles_Y + [mass_per_particle_cross]*N_particles_cross
frame.particles.moment_inertia = [0.0, 0.0, I_bar[2, 2]] * N_particles_bar + [0.0, 0.0, I_Y[2, 2]] * N_particles_Y +[0.0, 0.0, I_cross[2, 2]] * N_particles_cross # enforce 2D

frame.particles.orientation = np.vstack([ini_quaternions[bar_ids], add_quater, ini_quaternions[np.hstack([Y_ids, cross_ids])]])

#------------------ create new initial frame---------------------
if os.path.exists(folderprefix+f'Bar{barh0scale:.1f}_Y{Yh0scale:.1f}{suffix}.gsd'):
    os.remove(folderprefix+f'Bar{barh0scale:.1f}_Y{Yh0scale:.1f}{suffix}.gsd')
with gsd.hoomd.open(name=folderprefix+f'Bar{barh0scale:.1f}_Y{Yh0scale:.1f}{suffix}.gsd', mode='x') as f:
    f.append(frame)

simulation = hoomd.Simulation(device=gpu, seed=simrandseed)
simulation.create_state_from_gsd(filename=folderprefix+f'Bar{barh0scale:.1f}_Y{Yh0scale:.1f}{suffix}.gsd')
rigid = hoomd.md.constrain.Rigid()
rigid.body['Bar'] = {
    "constituent_types": ['M1_bar', 'M2_bar'],
    "positions": constituent_poss_bar,
    "orientations": constituent_orient_bar,
    }
rigid.body['Y'] = {
        "constituent_types": ['M1_Y', 'M2_Y', 'M3_Y'],
        "positions": constituent_poss_Y,
        "orientations": constituent_orient_Y,
        }
rigid.body['Cross'] = {
        "constituent_types": ['M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'],
        "positions": constituent_poss_cross,
        "orientations": constituent_orient_cross,
        }

rigid.create_bodies(simulation.state)
communicator = simulation.device.communicator
rank = communicator.rank
communicator.barrier_all()
cursnap = simulation.state.get_snapshot()
if rank==0:
    render_m2cm(cursnap, savefig=folderprefix+f'Bar{barh0scale:.1f}_Y{Yh0scale:.1f}{suffix}.png', Lx=None, Ly=None, exist_bar=plot_barid+1, exist_Y=plot_Yid+1, exist_cross=plot_crossid+1)

Lx, Ly = (simulation.state.box.Lx, simulation.state.box.Ly)
print(f'Lx: {Lx:.4f} m, Ly: {Ly:.4f} m.')

V_particle_bar = barl0_m*barh0scale*2*magr_m + np.pi*magr_m**2
V_particle_cross = (barl0_m*2*magr_m + np.pi*magr_m**2)*2-(2*magr_m)**2
V_particle_Y = np.pi*magr_m**2/2*3 + 3*halfl0_m*Yh0scale*2*magr_m - (3**0.5+1/3**0.5)*magr_m**2*3/4

print(f'Initial volume fraction: {((N_particles_bar*V_particle_bar + N_particles_cross*V_particle_cross + N_particles_Y*V_particle_Y) / simulation.state.box.volume):.4f}')
communicator.barrier_all()
print(f"Wait to start simulation by rank: {rank}.")

#--------------------------------------- [change integrator] ------------------------------
integrator = hoomd.md.Integrator(dt=thisdt, integrate_rotational_dof=True) # s
integrator.rigid = rigid
simulation.operations.integrator = integrator

#---------------------------------------------------------------------------------------------------
# pairwise potential
nl = hoomd.md.nlist.Cell(buffer=0, exclusions=['body'])
# mie
mie = hoomd.md.pair.Mie(nlist=nl, mode='shift')
mie.params[(['M1_bar', 'M2_bar', 'M1_Y', 'M2_Y', 'M3_Y', 'M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'],['M1_bar', 'M2_bar', 'M1_Y', 'M2_Y', 'M3_Y', 'M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'])] = dict(epsilon=0.6, sigma=49/50*2*magr_m, n=50, m=49) # esp=0.6
mie.r_cut[(['M1_bar', 'M2_bar', 'M1_Y', 'M2_Y', 'M3_Y', 'M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'],['M1_bar', 'M2_bar', 'M1_Y', 'M2_Y', 'M3_Y', 'M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'])] = 2*magr_m
# need to zero out interactions between other pair, otherwise, ERROR when add to integrator
mie.params[(['Bar', 'Y', 'Cross'], ['Bar', 'Y','Cross', 'M1_bar', 'M2_bar', 'M1_Y', 'M2_Y', 'M3_Y', 'M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'])] = dict(epsilon=0, sigma=0, n=50, m=49)
mie.r_cut[(['Bar', 'Y', 'Cross'], ['Bar', 'Y', 'Cross', 'M1_bar', 'M2_bar', 'M1_Y', 'M2_Y', 'M3_Y', 'M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'])] = 0
integrator.forces.append(mie)

# dipole
bar_dipole_1 = np.pi/6
bar_dipole_2 = np.pi*7/6

# dummy
cross_dipole_1 = 0
cross_dipole_2 = 0
cross_dipole_3 = 0
cross_dipole_4 = 0

Y_dipole_1 = np.pi/6
Y_dipole_2 = np.pi/6
Y_dipole_3 = np.pi/6

magdipole = hoomd.md.pair.aniso.Dipole(nl, default_r_cut=r_mag_cutoff) # <---
magdipole.params[(['M1_bar', 'M2_bar', 'M1_Y', 'M2_Y', 'M3_Y', 'M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'],['M1_bar', 'M2_bar', 'M1_Y', 'M2_Y', 'M3_Y', 'M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'])] = dict(A=mu0/(4*np.pi), kappa=0.0)
# the magnetic magnitude of the particle local reference frame as a tuple
magdipole.mu['M1_bar'] = (np.cos(bar_dipole_1)*mag_amp, np.sin(bar_dipole_1)*mag_amp, 0.0*mag_amp) 
magdipole.mu['M2_bar'] = (np.cos(bar_dipole_2)*mag_amp, np.sin(bar_dipole_2)*mag_amp, 0.0*mag_amp) 
magdipole.mu['M1_Y'] = (np.cos(Y_dipole_1)*mag_amp, np.sin(Y_dipole_1)*mag_amp, 0.0*mag_amp) 
magdipole.mu['M2_Y'] = (np.cos(Y_dipole_2)*mag_amp, np.sin(Y_dipole_2)*mag_amp, 0.0*mag_amp) 
magdipole.mu['M3_Y'] = (np.cos(Y_dipole_3)*mag_amp, np.sin(Y_dipole_3)*mag_amp, 0.0*mag_amp) 
magdipole.mu['M1_cross'] = (np.cos(cross_dipole_1)*mag_amp, np.sin(cross_dipole_1)*mag_amp, 0.0*mag_amp) 
magdipole.mu['M2_cross'] = (np.cos(cross_dipole_2)*mag_amp, np.sin(cross_dipole_2)*mag_amp, 0.0*mag_amp) 
magdipole.mu['M3_cross'] = (np.cos(cross_dipole_3)*mag_amp, np.sin(cross_dipole_3)*mag_amp, 0.0*mag_amp) 
magdipole.mu['M4_cross'] = (np.cos(cross_dipole_4)*mag_amp, np.sin(cross_dipole_4)*mag_amp, 0.0*mag_amp) 

magdipole.mu['Bar'] = (0, 0, 0) 
magdipole.mu['Y'] = (0, 0, 0) 
magdipole.mu['Cross'] = (0, 0, 0) 
magdipole.params[(['Bar', 'Y', 'Cross'], ['Bar', 'Y', 'Cross', 'M1_bar', 'M2_bar', 'M1_Y', 'M2_Y', 'M3_Y', 'M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'])] = dict(A=0, kappa=0.0)
magdipole.r_cut[(['Bar', 'Y', 'Cross'], ['Bar', 'Y', 'Cross', 'M1_bar', 'M2_bar', 'M1_Y', 'M2_Y', 'M3_Y', 'M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'])] = 0
integrator.forces.append(magdipole)

# ALJ
lam = 2**(1/6)
scaling = 0.99
sig = scaling*2*magr_m/lam
beta = 0.05
rradii = magr_m - beta*magr_m
aljscale = max(barh0scale, Yh0scale)
alj_r_cut = (aljscale*halfl0_m+magr_m)*2*1.1
alj = hoomd.md.pair.aniso.ALJ(nl, default_r_cut=alj_r_cut) # [change, special care]
alj.params[(['M1_bar', 'M2_bar', 'M1_Y', 'M2_Y', 'M3_Y', 'M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'],['M1_bar', 'M2_bar', 'M1_Y', 'M2_Y', 'M3_Y', 'M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'])] = dict(epsilon=0.03, sigma_i=sig, sigma_j=sig, alpha=0,
                                            contact_ratio_i=beta, contact_ratio_j=beta, 
                                            average_simplices=True)
alj.params[(['Bar', 'Y', 'Cross'], ['Bar', 'Y', 'Cross', 'M1_bar', 'M2_bar', 'M1_Y', 'M2_Y', 'M3_Y', 'M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'])] = dict(epsilon=0.0, sigma_i=0.0, sigma_j=0.0, alpha=0,)
alj.r_cut[(['Bar', 'Y', 'Cross'], ['Bar', 'Y', 'Cross', 'M1_bar', 'M2_bar', 'M1_Y', 'M2_Y', 'M3_Y', 'M1_cross', 'M2_cross', 'M3_cross', 'M4_cross'])] = 0
alj.shape["M1_bar"] = dict(vertices=[(0,0,0), (halfl0_m*barh0scale,0,0)],
                    rounding_radii=rradii,
                    faces=[])
alj.shape["M2_bar"] = dict(vertices=[(0,0,0), (-halfl0_m*barh0scale,0,0)],
                    rounding_radii=rradii,
                    faces=[])
alj.shape["Bar"] = dict(vertices=[(0,0,0)],
                    rounding_radii=magr_m,
                    faces=[])

alj.shape["M1_Y"] = dict(vertices=[(0,0,0), (-halfl0_m*Yh0scale,0,0)],
                    rounding_radii=rradii,
                    faces=[])
# M2 is defined as no. 2 bar, then rotates about vetice (0,0,0) for 120 deg CW, then is mounted to M2 position
alj.shape["M2_Y"] = dict(vertices=[(0,0,0), (-halfl0_m*Yh0scale,0,0)],
                    rounding_radii=rradii,
                    faces=[])
# M3 is defined as no. 3 bar, then rotates about vetice (0,0,0) for 120 deg CCW , then is mounted to M3 position
alj.shape["M3_Y"] = dict(vertices=[(0,0,0), (-halfl0_m*Yh0scale,0,0)],
                    rounding_radii=rradii,
                    faces=[])
alj.shape["Y"] = dict(vertices=[(0,0,0)],
                    rounding_radii=magr_m,
                    faces=[])

alj.shape["M1_cross"] = dict(vertices=[(0,0,0), (halfl0_m,0,0)],
                    rounding_radii=rradii,
                    faces=[])
alj.shape["M2_cross"] = dict(vertices=[(0,0,0), (-halfl0_m,0,0)],
                    rounding_radii=rradii,
                    faces=[])
alj.shape["M3_cross"] = dict(vertices=[(0,0,0), (halfl0_m,0,0)],
                    rounding_radii=rradii,
                    faces=[])
alj.shape["M4_cross"] = dict(vertices=[(0,0,0), (-halfl0_m,0,0)],
                    rounding_radii=rradii,
                    faces=[])
alj.shape["Cross"] = dict(vertices=[(0,0,0)],
                    rounding_radii=magr_m,
                    faces=[])

integrator.forces.append(alj)

#---------------------------------------
# initialize 
simulation.run(0)

magene = magdipole.energy
aljene = alj.energy
if rank==0:
    print(f"Bar{barh0scale:.1f}_Y{Yh0scale:.1f}:")
    print(f'    Total mag energy:{magene} J.')
    print(f'    Total ALJ energy:{aljene} J.')
