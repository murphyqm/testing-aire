#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 29/10/2021
by murphyqm

"""
import pathlib
import sys
import pandas as pd
import numpy as np

import pytesimint.define_matrix as dm
import pytesimint.effective_diffusivity as ed
import pytesimint.iteration as it


def set_grid_values_3d_rounded2(x, y, z,
                                blank_vol,
                                d, e, f,
                                a, b, c,
                                int_value,
                                mant_value):
    new_vol = np.full_like(blank_vol, mant_value)
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                if (((x[i] - d)**2)/(a**2)) + \
                 (((y[j] - e)**2)/(b**2)) + \
                 (((z[k] - f)**2)/(c**2)) <= 1:
                    new_vol[i, j, k] = int_value
    return new_vol


def to_3d(bg_T, Nx, Ny, Nz):
    bg_3d = np.broadcast_to(bg_T, (Nx, Ny, Nz))
    return bg_3d


value = int(sys.argv[1])
param_file = sys.argv[2]
parent_folder = sys.argv[3]

# print(value, param_file, parent_folder)

params = pd.read_csv(param_file)

# set boundary condition
boundary_cond = params["boundary_cond"][value]

# set number of grid points
Nx = params["Nx"][value]
Ny = params["Ny"][value]
Nz = params["Nz"][value]

# Set size of grid
Lx = params["Lx"][value]
Ly = params["Ly"][value]
Lz = params["Lz"][value]

# Check the spacing of the grid
dx = dm.spacing(Nx, Lx)
dy = dm.spacing(Ny, Ly)
dz = dm.spacing(Nz, Lz)
# print(dx, dy, dz)

# Now define a 3D mesh
x, y, z, blank_vol = dm.define_grid_3d(Nx, Ny, Nz, Lx, Ly, Lz)

# to define where is inside and outside the intrusion
interior = 1
exterior = 0

# setting up initial values for temp and diff
int_temp = params["int_temp"][value]
# ext_temp = params["ext_temp"][value]

# setting up background temperature gradient
top_T = params["temp_top"][value]
bottom_T = params["temp_bottom"][value]
bg_T = np.linspace(top_T, bottom_T, Nx)
ext_temp = to_3d(bg_T, Nx, Ny, Nz)

# initial diffusivities
# int_diff = params["int_diff"][value]
# ext_diff = params["ext_diff"][value]
int_diff = 3.774660434749065e-06
ext_diff = 1.0963794262207911e-06

# contacts (need to also add this to the param file
# x1 = params["x1"][value]
# x2 = params["x2"][value]
# y1 = params["y1"][value]
# y2 = params["y2"][value]
# z1 = params["z1"][value]
# z2 = params["z2"][value]

# define the origin of the ellipsoid
x_mid = params["x_mid"][value]
y_mid = params["y_mid"][value]
z_mid = params["z_mid"][value]

# define the radii of the ellipsoid
r_x = params["r_x"][value]
r_y = params["r_y"][value]
r_z = params["r_z"][value]

# defining where is inside and outside the intrusion
location_of_intrusion = dm.set_grid_values_3d_rounded(x, y, z, blank_vol,
                                                      x_mid, y_mid, z_mid,
                                                      r_x, r_y, r_z,
                                                      interior, exterior)

# setting initial conditions
initial_diffs = dm.set_grid_values_3d_rounded(x, y, z, blank_vol,
                                              x_mid, y_mid, z_mid,
                                              r_x, r_y, r_z,
                                              int_diff, ext_diff)
initial_temps = set_grid_values_3d_rounded2(x, y, z, blank_vol,
                                            x_mid, y_mid, z_mid,
                                            r_x, r_y, r_z,
                                            int_temp, ext_temp)

# Instantiate theta function object
T_L = params["T_L"][value]
T_S = params["T_S"][value]

freezing_func = ed.FreezingFunction(T_L, T_S)
print("instantiate freezing_func")

metal_fraction = params["metal_fraction"][value]
cond_metal_s = params["cond_metal_s"][value]
cond_metal_l = params["cond_metal_l"][value]
cond_olivine = params["cond_olivine"][value]
dens_liq_metal = params["dens_liq_metal"][value]
dens_solid_metal = params["dens_solid_metal"][value]
dens_olivine = params["dens_olivine"][value]
heat_cap_liq_metal = params["heat_cap_liq_metal"][value]
heat_cap_solid_metal = params["heat_cap_solid_metal"][value]
heat_cap_ol = params["heat_cap_ol"][value]
latent_heat = params["latent_heat"][value]
print("Loaded params")

# instantiate apparent diffusivity object
app_diff = ed.nnnAppDiff(metal_fraction, cond_metal_s, cond_metal_l, cond_olivine,
                         dens_liq_metal, dens_solid_metal, dens_olivine,
                         heat_cap_liq_metal, heat_cap_solid_metal,
                         heat_cap_ol, latent_heat, freezing_func)
print("Instantiated diff object.")
# set the timestep
dt = params["dt"][value]

# set the run ID
runID = params["id"][value]
check_path = f"{parent_folder}{runID}"
pathlib.Path(check_path).mkdir(parents=True, exist_ok=True)
folder = f"{check_path}/"

# set number of iterations and save frequency
iterations = params["iterations"][value]
save_iter = params["save_iter"][value]
which_iter = params["which_iter"][value]

if which_iter == "v8":
    iteration_list = params["iteration_list"][value]

print("starting iterations now.")
if which_iter == "v4":
    print("Using old iterator, v4.")
    it.v4_iter_func(initial_temps,
                    initial_diffs,
                    location_of_intrusion,
                    app_diff,
                    boundary_cond,
                    x, y, z,
                    dx, dy, dz,
                    Nx, Ny, Nz,
                    dt,
                    folder,
                    iterations=iterations,
                    save_iter=save_iter, 
                    fileID=runID)

if which_iter == "v7":
    print("Using new iterator, v7.")
    it.v7_iter_func(initial_temps,
                    initial_diffs,
                    location_of_intrusion,
                    app_diff,
                    boundary_cond,
                    x, y, z,
                    dx, dy, dz,
                    Nx, Ny, Nz,
                    dt,
                    folder,
                    iterations=iterations,
                    save_iter=save_iter, 
                    fileID=runID)
print("finished iterations")

if which_iter == "v8":
    print("Using new iterator, v8.")
    it.v8_iter_func(initial_temps,
                    initial_diffs,
                    location_of_intrusion,
                    app_diff,
                    boundary_cond,
                    x, y, z,
                    dx, dy, dz,
                    Nx, Ny, Nz,
                    dt,
                    folder,
                    iterations=iterations,
                    save_iter=save_iter, 
                    fileID=runID,)
print("finished iterations")
