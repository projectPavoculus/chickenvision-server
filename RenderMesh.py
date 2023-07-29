

# -*- coding: utf-8 -*-
"""
Handling .obj mesh!
"""

import open3d as o3d
import math
import numpy as np


HOMEDIR = "X:/Testing" # set this to wherever you have the obj file
OBJPATH = "ChickenHat"+".obj"
SAVEPATH = "output"+"png"


def Radians(degrees):
    return tuple([math.radians(x) for x in degrees])


def Render(savePath, objPath, angles, xyz, view_w, view_h):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=view_w, height=view_h)
    vis.get_render_option().background_color = np.array([1.0, 1.0, 1.0])  # transparent background
    
    
    
    angles = Radians(angles) # translating into radians
    mesh = o3d.io.read_triangle_mesh(objPath) # reading objets
    mesh.compute_vertex_normals()
    
    R = mesh.get_rotation_matrix_from_axis_angle(angles)
    mesh.rotate(R, center=(0, 0, 0))
    
    mesh.translate(xyz)
    
    
    vis.add_geometry(mesh)
    
    
    viewControl = vis.get_view_control()
    viewControl.set_lookat([0,0,0])
    viewControl.set_zoom(3)

    vis.update_geometry(mesh)
    vis.poll_events() # need to have these two lines to work
    vis.update_renderer() # need to have these two lines to work
    
    vis.capture_screen_image(savePath) #saving
    #vis.destroy_window()
    vis.run()




# For trial load an .obj file and try this:
Render(SAVEPATH, OBJPATH, (45,45,45), [0,0, 100], 1920, 1080)
