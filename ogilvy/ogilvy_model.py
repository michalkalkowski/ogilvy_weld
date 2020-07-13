#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 10:44:02 2018

Implementation of the Ogilvy's geometrical model for grain orientations in
austenitic steel welds. Based on:
[1] Ogilvy, J.A., 1985. Computerized ultrasonic ray tracing in austenitic steel. NDT International 18, 67–77. https://doi.org/10.1016/0308-9126(85)90100-2

author: Michal K Kalkowski, m.kalkowski@imperial.ac.uk
Copyright (C) 2018 Michal K Kalkowski (MIT License)
"""
import operator
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

class Ogilvy_weld(object):
    """
    Ogilvy model object collecting parameters of a weld and enabling to calculate and visualise grain orientations.

    On initialisation, one should set up weld parameters:
    ---
    a: thickness
    b: weld base
    c: length of the top surface
    T : float, positive weld parameter (proportional to the tangents of
            orientations at the grain boundary)
    n_weld : float, determines the rate of change of the angle along z-axis

    """
    def __init__(self, weld_parameters, **kwargs):
        for parameter, value in weld_parameters.items():
            setattr(self, parameter, value)
        for key, value in kwargs:
            setattr(self, key, value)
        self.xx, self.yy = None, None
        self.mesh_x, self.mesh_y = None, None
        self.centro_x, self.centro_y = None, None
        self.weld_angle = np.arctan((self.c - self.b)/(2*self.a))

        # check if all parameters are there
        try:
            print('------Ogilvy model setup------')
            print('Tangent parameter T: {}'.format(self.T))
            print('Rate of change along the z-axis: {}'.format(self.n_weld))
            print('Weld thickness: {}'.format(self.a))
            print('Chamfer base: {}'.format(self.b))
            print('Chamfer top: {}'.format(self.c))
            print('----------------------------')
        except AttributeError:
            print("Ogilvy parameters corrupt. Please redefine the model.")

    def define_grid_size(self, size, use_centroids=True,
                         add_boundary_cells=True, boundary_offset=1):
        """
        Defines the size of the grid for the MINA model.

        Parameters:
        ---
        size: float, grid size in milimetres
        use_centroids: bool, determines whether MINA grid points are understood
                       as nodes or as centroids of cells (default)
        add_boundary_cells: bool, determines whether MINA cells close to the
                            boundary, with with the centroid outside of the
                            weld are considered (default: True).
        """

        self.use_centroids = use_centroids
        self.grid_size = float(size)
        elements_x = int(np.ceil(self.c/self.grid_size))
        elements_y = int(np.ceil(self.a/self.grid_size))
        # Set up the grid of points
        seeds_x = np.arange(0, (elements_x + 1)*self.grid_size, self.grid_size)
        seeds_y = np.arange(0, (elements_y + 1)*self.grid_size, self.grid_size)
        self.xx, self.yy = np.meshgrid(seeds_x, seeds_y)
        self.xx += -self.c/2.
        # Calculate the centroids
        self.centro_x = ((self.xx[:, :-1] + self.xx[:, 1:])/2)[:-1, :]
        self.centro_y = ((self.yy[:-1, :] + self.yy[1:, :])/2)[:, :-1]
        if use_centroids:
            self.in_weld = np.zeros(self.centro_x.shape)
            self.in_weld[np.where((self.centro_y >=
                                   self.get_chamfer_profile(self.centro_x))
                                  & (self.centro_y <= self.a + self.grid_size/2))] = 1
            self.mesh_x = self.centro_x
            self.mesh_y = self.centro_y
        else:
            self.in_weld = np.zeros(self.xx.shape)
            self.in_weld[np.where((self.yy >=
                                   self.get_chamfer_profile(self.xx))
                                  & (self.yy <= self.a + self.grid_size/2))] = 1
            self.mesh_x = self.xx
            self.mesh_y = self.yy

        # Seeding defined above ignores some cells close to the weld boundary.
        # The code below adds the cells which are crossed by the boundary and
        # the corners.
        # Determine the the cells that encompass weld boundary
        A = np.array([self.b/2, 0]).reshape(-1, 1)
        B = np.array([self.c/2, self.a]).reshape(-1, 1)
        mina_grid = np.c_[self.centro_x.flatten(),
                          self.centro_y.flatten()]
        d_right = (np.cross(B - A, mina_grid.T - A, axis=0)
                   / np.linalg.norm(B - A)).reshape(self.centro_x.shape)
        A = np.array([-self.b/2, 0]).reshape(-1, 1)
        B = np.array([-self.c/2, self.a]).reshape(-1, 1)
        d_left = (np.cross(B - A, mina_grid.T - A, axis=0)
                  / np.linalg.norm(B - A)).reshape(self.centro_x.shape)
        d = np.c_[d_left[:, :int(self.centro_x.shape[1]/2)],
                  d_right[:, int(self.centro_x.shape[1]/2):]]
        critical = (np.sin(np.pi/4 + np.arctan((self.c - self.b)/(2*self.a)))
                    * boundary_offset*self.grid_size*2**0.5/2)
        self.boundary_cells = np.zeros(self.mesh_x.shape, dtype=bool)
        self.boundary_cells[np.where((abs(d) <= critical))] = True
        # Add boundary cells to weld for ray tracing
        if add_boundary_cells:
            self.in_weld[self.boundary_cells] = 1

        self.mesh_x_strict = np.copy(self.mesh_x)
        self.mesh_y_strict = np.copy(self.mesh_y)
        self.mesh_x_strict[self.in_weld == 0] = np.nan
        self.mesh_y_strict[self.in_weld == 0] = np.nan


    def get_chamfer_profile(self, x):
        """
        Outputs the profile of the weld chamfer (a simple mapping from the
        horizontal coordinate(x) to the vertical coordinate(y).
        Used for checking whether a certain point is inside the weld or not.

        Parameters:
        ---
        y: float, horizontal coordinate)
        """
        boundary_gradient = 2*self.a/(self.c - self.b)
        f = boundary_gradient*(abs(x) - self.b/2)
        f *= (f >= 0)
        return f


    def solve(self, y=None, z=None):
        """Solves the Ogilvy's model."""
        self.alpha = np.arctan((self.c - self.b)/2/self.a)
        if y is None and z is None:
            if self.mesh_x is None:
                self.define_grid_size(2)
                print('Using a default grid of 2 mm...')
            theta = np.arctan(self.T*abs(self.b/2 +
                self.mesh_y*np.tan(self.alpha))/abs(self.mesh_x)**self.n_weld)
            theta[self.mesh_x >= 0] *= -1
            # NaN grid points not in weld
            not_in_weld = self.mesh_y < self.get_chamfer_profile(self.mesh_x)
        else:
            if type(y) is not np.ndarray:
                y = np.array([[y]])
                z = np.array([[z]])
            theta = np.arctan(self.T*abs(self.b/2 + z*np.tan(self.alpha))/abs(y)**self.n_weld)
            theta[y >= 0] *= -1
            # NaN grid points not in weld
            not_in_weld = z < self.get_chamfer_profile(y)
#        theta[not_in_weld] = np.nan
        self.grain_orientations = theta - np.pi/2
        self.grain_orientations[self.grain_orientations < -np.pi/2] += np.pi
        self.grain_orientations[self.grain_orientations > np.pi/2] -= np.pi
        self.grain_orientations[not_in_weld] = np.nan
        self.mina_grid_points = np.c_[self.mesh_x_strict.flatten(),
                                 self.mesh_y_strict.flatten()]
        self.grain_orientations_full = np.copy(self.grain_orientations)
        self.grain_orientations = self.grain_orientations[self.in_weld == 1].flatten()

        # self.grain_orientations = self.grain_orientations[~np.isnan(self.grain_orientations)]
        self.mina_grid_points = self.mina_grid_points[~np.isnan(self.mina_grid_points).any(axis=1)]
        # cells_above = np.where(self.mina_grid_points[:, 1] > self.a)[0]
        # self.grain_orientations = np.delete(self.grain_orientations,
        #                                     cells_above)
        # self.mina_grid_points = np.delete(self.mina_grid_points, cells_above,
        #                                   axis=0)
        # Create a KDTree for quick point lookup
        self.tree = cKDTree(self.mina_grid_points)
        self.grid = cKDTree(np.c_[self.mesh_x.flatten(),
                                  self.mesh_y.flatten()])
        # in case any nans occur (cell not assigned to a pass, etc. may happen
        # close to boundary), we fill them with the mean of surrounding cells
        nan_cells = np.where(np.isnan(self.grain_orientations))[0]
        if len(nan_cells) > 0:
            for nan_cell in nan_cells:
                # Find neighbouring cells
                distance, index = self.tree.query(
                    self.mina_grid_points[nan_cell], 6)
                neighbours = self.grain_orientations[index[distance ==
                                                           self.grid_size]]
                mean_orient = neighbours[~np.isnan(neighbours)].mean()
                self.grain_orientations[nan_cell] = mean_orient
        nan_cells = np.where(np.isnan(self.grain_orientations_full) &
                             (self.in_weld == 1))
        if len(nan_cells[0]) > 0:
            for nan_cell in range(len(nan_cells[0])):
                # Find neighbouring cells
                this_cell = nan_cells[0][nan_cell], nan_cells[1][nan_cell]
                neighbours = [[this_cell[0] + i
                               for i in [1, 1, 0, -1, -1, -1, 0, 1]],
                              [this_cell[1] + j
                              for j in [0, 1, 1, 1, 0, -1, -1, -1]]]
                ind = np.array(neighbours)
                ind = ind[:, (ind[0] < self.grain_orientations_full.shape[0]) &
                (ind[1] < self.grain_orientations_full.shape[1])].tolist()
                neighbours = self.grain_orientations_full[tuple(ind)]
                mean_orient = np.nanmean(neighbours)
                self.grain_orientations_full[nan_cells[0][nan_cell],
                                            nan_cells[1][nan_cell]] = mean_orient
 

    def get_angle(self, location):
        """
        Returns the orientation angle  corresponding to the grid point that
        encompasses current location.

        Parameters:
        ---
        location: ndarray, shape: (3,) an array with coordinates of the location of
                    interest

        Returns:
        ---
        angle: float, orientation angle (with respect to the vertical.
        """
        self.solve(y=location[1], z=location[2])
        return self.grain_orientations.squeeze()

    def get_ficticious_interface_angle(self, coordinates):

        # Express the angle wqith reference to horizontal
        orientation = self.get_angle(coordinates) + np.pi/2
        interface_angle = np.arctan(-1/self.T*np.tan(orientation)/np.tan(self.alpha))
        if interface_angle < 0:
            interface_angle += np.pi
        return interface_angle

    def plot_grain_orientations(self, scale=2, grid=False, ax=None, alt=None,
                                color='black'):
        """
        Plots final grain orientations.

        Parameters:
        ---
        scale: float, scale parameter for the plt.quiver command.
               Smaller scale - larger arrows and vice versa
        grid: boolean, if True, the MINA grid is also plotted.
        ax: handle to existing axes object, if plotting into already existing
            axes, None otherwise (default).
        alt: ndarray, an array og orientations to replace those which are the
             attribute, useful only for debugging. None if not desired (default)
        color: str, color of orientation lines, default 'black'.
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        if grid:
            if self.use_centroids:
                ax.plot(self.xx, self.yy, lw=0.5, c='lightgray')
                ax.plot(self.xx.T, self.yy.T, lw=0.5, c='lightgray')
            else:
                ax.plot(self.xx, self.yy, 'o', ms=2, c='lightgray')
        if alt is not None:
            orientations = alt
        else:
            orientations = self.grain_orientations.flatten()
        starters = np.array([
            self.mina_grid_points[:, 0] - scale/2*np.cos(orientations
                                                         + np.pi/2),
            self.mina_grid_points[:, 1] - scale/2*np.sin(orientations
                                                         + np.pi/2)])
        ends = np.array([
            self.mina_grid_points[:, 0] + scale/2*np.cos(orientations
                                                         + np.pi/2),
            self.mina_grid_points[:, 1] + scale/2*np.sin(orientations
                                                         + np.pi/2)])

        ax.plot([starters[0, :], ends[0, :]], [starters[1, :], ends[1, :]],
                c=color)
        ax.set_aspect('equal')
        ax.set_title('grain orientations')
        ax.plot([-self.b/2, -self.c/2], [0, self.a], lw=0.5, c='gray')
        ax.plot([self.b/2, self.c/2], [0, self.a], lw=0.5, c='gray')
        ax.plot([-1.2*self.c/2, 1.2*self.c/2], [0, 0], lw=0.5, c='gray')
        ax.plot([-1.2*self.c/2, 1.2*self.c/2], [self.a, self.a], lw=0.5,
                c='gray')
        ax.set_xlabel(r'$x$ in mm')
        ax.set_ylabel(r'$y$ in mm')
        plt.tight_layout()
        plt.show()
