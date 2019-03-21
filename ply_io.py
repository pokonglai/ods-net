import sys
import os

import numpy as np
import cv2
import scipy.io

from utils import *


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def compute_true_normal_map(img):
    ''' Compute the normal vector using the 3d point cloud rather than the grid structure of the depth map. '''
    rows, cols = img.shape
    normap = np.zeros((rows, cols, 3), np.float32)
    
    for i in range(0, rows):
        for j in range(0, cols):
            ij_3dpos, nvec_nonorm = compute_true_normal(img, i,j)
            nvec=normalize(nvec_nonorm)

            ## flip orientation according to distance with origin
            pos_plus = nvec
            pos_minus = -1.0*nvec

            ray_to_origin = (ij_3dpos / np.linalg.norm(ij_3dpos))
            plus_angle = np.arccos(np.clip(np.dot(pos_plus, ray_to_origin), -1.0, 1.0))
            minus_angle = np.arccos(np.clip(np.dot(pos_minus, ray_to_origin), -1.0, 1.0))
            if plus_angle > minus_angle: nvec = -1.0*nvec

            normap[i][j] = nvec
    return normap

def compute_true_3dpos(img_depth, r, c):
    ''' Given a depth map and a row and col, return the true 3d position. '''
    img_h, img_w = img_depth.shape

    depth_val = img_depth[r, c]

    if depth_val == 0: return np.array([0,0,0])

    w_ratio = c/img_w # width is the longitude
    w_angle = np.deg2rad(w_ratio*360)

    h_ratio = r/img_h # height is the latitude 
    h_angle = np.deg2rad((h_ratio-0.5)*180)

    x = 1 * np.cos(h_angle) * np.cos(w_angle)
    y = 1 * np.sin(h_angle)
    z = 1 * np.cos(h_angle) * np.sin(w_angle)

    ptvector = np.array([x, y, z])
    unit_ptvector = ptvector/np.linalg.norm(ptvector) # normalize vector 
    unit_ptvector *= depth_val # extend the unit vector to the approriate depth
    return unit_ptvector

def compute_true_normal(img_depth, r, c):
    ''' Compute the normal vector given a depth map and a row and col. '''
    rows, cols = img_depth.shape
    if (r == 0) and (c == 0):
        p1 = compute_true_3dpos(img_depth, r, c)
        p2 = compute_true_3dpos(img_depth, r, c+1)
        p3 = compute_true_3dpos(img_depth, r+1, c)
    elif (r == 0) and (c != 0): 
        p1 = compute_true_3dpos(img_depth, r, c)
        p2 = compute_true_3dpos(img_depth, r, c-1)
        p3 = compute_true_3dpos(img_depth, r+1, c)
    elif (r != 0) and (c == 0):
        p1 = compute_true_3dpos(img_depth, r, c)
        p2 = compute_true_3dpos(img_depth, r, c+1)
        p3 = compute_true_3dpos(img_depth, r-1, c)
    else:
        p1 = compute_true_3dpos(img_depth, r, c)
        p2 = compute_true_3dpos(img_depth, r-1, c)
        p3 = compute_true_3dpos(img_depth, r, c-1)

    return p1, normal_from_3point(p1, p2, p3)


def normal_from_3point(p1, p2, p3):
    v1 = p3 - p1
    v2 = p2 - p1
    return np.cross(v1, v2) # the cross product is a vector normal to the plane


def images_to_omni_pointcloud_equi_rectangular_vectorized(img_color, img_depth, hasNormals=False, img_normals=None):
    """ Given an rgb-d image pair, return a point cloud using the spherical model. Much faster as we use vectorized NumPy operations. """
    img_h, img_w = img_depth.shape

    angles_long = np.linspace(np.deg2rad(0), np.deg2rad(360), img_w)
    angles_lat = np.linspace(np.deg2rad(-90), np.deg2rad(90), img_h)

    longv, latv = np.meshgrid(angles_long, angles_lat)
    x = 1 * np.cos(latv) * np.cos(longv)
    y = 1 * np.sin(latv)
    z = 1 * np.cos(latv) * np.sin(longv)

    reds = img_color[...,0].flatten()
    greens = img_color[...,1].flatten()
    blues = img_color[...,2].flatten()

    xvals = x.flatten()
    yvals = y.flatten()
    zvals = z.flatten()

    ptcloud = np.column_stack([xvals, yvals, zvals])
    ptcloud = np.divide(ptcloud, np.linalg.norm(ptcloud, axis=1)[:, None]) # normalize all points

    depth_vals = img_depth.flatten()
    ptcloud = np.multiply(ptcloud, depth_vals[:, None]) ## now extend them to their approriate distance

    # rgb_vals = np.column_stack([blues/255, greens/255, reds/255, np.ones_like(reds)])
    rgb_vals = np.column_stack([blues, greens, reds])
    return ptcloud, rgb_vals


def images_to_omni_pointcloud_equi_rectangular(img_color, img_depth, hasNormals=False, img_normals=None):
    """ Given an rgb-d image pair, return a point cloud using the spherical model. """
    ptcloud = []

    img_h, img_w = img_depth.shape
    n_points_total = img_w*img_h
    n_points_processed = 0

    for h in range(0, img_h):
        for w in range(0, img_w):
            depth_val = img_depth[h, w]
            # b, g, r = img_color[h, w]
            r, g, b = img_color[h, w]
            r = (int)(255*r)
            g = (int)(255*g)
            b = (int)(255*b)

            w_ratio = w/img_w # width is the longitude
            w_angle = np.deg2rad(w_ratio*360)

            h_ratio = h/img_h # height is the latitude 
            h_angle = np.deg2rad((h_ratio-0.5)*180)


            x = 1 * np.cos(h_angle) * np.cos(w_angle)
            y = 1 * np.sin(h_angle)
            z = 1 * np.cos(h_angle) * np.sin(w_angle)

            ptvector = np.array([x, y, z])
            unit_ptvector = ptvector/np.linalg.norm(ptvector) # normalize vector 
            unit_ptvector *= depth_val # extend the unit vector to the approriate depth

            if hasNormals: 
                nx, ny, nz = img_normals[h,w]
                ptcloud.append([unit_ptvector[0], unit_ptvector[1], unit_ptvector[2], r, g, b, nx, ny, nz])
            else: ptcloud.append([unit_ptvector[0], unit_ptvector[1], unit_ptvector[2], r, g, b])

            n_points_processed+=1
            print_text_progress_bar(n_points_processed/n_points_total, bar_name='Omni-images to point cloud   ')
    print()

    return np.array(ptcloud), n_points_total

def output_pointcloud(nVertices, ptcloud, strOutputPath):
    """ Given a point cloud produced from image_fusion, output it to a PLY file. """
    # open the file and write out the standard ply header

    nprops = ptcloud.shape[1] # number of point cloud property values (ie: color, normals)

    print("Writing point cloud to '" + str(strOutputPath + ".ply") + "' ... ", end="", flush=True)
    outputFile = open(strOutputPath + ".ply", "w")
    outputFile.write("ply\n")
    outputFile.write("format ascii 1.0\n")
    outputFile.write("comment generated via python script Process3DImage\n")
    outputFile.write("element vertex %d\n" %(nVertices))
    if nprops == 6:
        outputFile.write("property float x\n")
        outputFile.write("property float y\n")
        outputFile.write("property float z\n")
        outputFile.write("property uchar red\n")
        outputFile.write("property uchar green\n")
        outputFile.write("property uchar blue\n")
    elif nprops == 9:
        outputFile.write("property float x\n")
        outputFile.write("property float y\n")
        outputFile.write("property float z\n")
        outputFile.write("property float nx\n")
        outputFile.write("property float ny\n")
        outputFile.write("property float nz\n")
        outputFile.write("property uchar red\n")
        outputFile.write("property uchar green\n")
        outputFile.write("property uchar blue\n")
    outputFile.write("element face 0\n")
    outputFile.write("property list uchar int vertex_indices\n")
    outputFile.write("end_header\n")

    # output the actual points
    for pt in ptcloud:
        dx, dy, dz = pt[0:3]
        r, g, b = pt[3:6]
        if nprops == 6: outputFile.write("%10.6f %10.6f %10.6f %d %d %d\n" %(dx, dy, dz, r, g, b))
        elif nprops == 9:
            nx, ny, nz = pt[6:9]
            outputFile.write("%10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %d %d %d\n" %(dx, dy, dz, nx, ny, nz, r, g, b))

    outputFile.close()

    print("Finished!")

def images_to_omni_textured_mesh_equi_rectangular(img_color, img_depth, tri_dist_thres=0.75, hasNormals=False, img_normals=None):
    """ Given an rgb-d image pair, return a textured mesh using the spherical model. """
    ptcloud = []
    triangles = []
    texcoords = []

    img_h, img_w = img_depth.shape
    n_points_total = img_w*img_h
    n_points_processed = 0

    for h in range(0, img_h):
        for w in range(0, img_w):
            depth_val = img_depth[h, w]

            # b, g, r = img_color[h, w]
            r, g, b = img_color[h, w]

            w_ratio = w/img_w # width is the longitude
            w_angle = np.deg2rad(w_ratio*360)

            h_ratio = h/img_h # height is the latitude 
            h_angle = np.deg2rad((h_ratio-0.5)*180)


            x = 1 * np.cos(h_angle) * np.cos(w_angle)
            y = 1 * np.sin(h_angle)
            z = 1 * np.cos(h_angle) * np.sin(w_angle)

            ptvector = np.array([x, y, z])
            unit_ptvector = ptvector/np.linalg.norm(ptvector) # normalize vector 
            unit_ptvector *= depth_val # extend the unit vector to the approriate depth

            if hasNormals: 
                nx, ny, nz = img_normals[h,w]
                ptcloud.append([unit_ptvector[0], unit_ptvector[1], unit_ptvector[2], r, g, b, nx, ny, nz])
            else: ptcloud.append([unit_ptvector[0], unit_ptvector[1], unit_ptvector[2], r, g, b])

            texcoords.append([1.0 - w_ratio, 1.0 - h_ratio])


            n_points_processed+=1
            print_text_progress_bar(n_points_processed/n_points_total, bar_name='Omni-images to textured mesh   ')
    print()
 
    n_triangles_processed = 0
    for h in range(0, img_h):
        for w in range(0, img_w):
            idx_00 = h*img_w + w         ## top left
            idx_10 = h*img_w + w + 1     ## top right
            idx_01 = (h+1)*img_w + w     ## bottom left
            idx_11 = (h+1)*img_w + w + 1 ## bottom right

            if (idx_00 < len(ptcloud)) and (idx_10 < len(ptcloud)) and (idx_01 < len(ptcloud)) and (idx_11 < len(ptcloud)):
                pt_tl = np.array(ptcloud[idx_00][0:3])
                pt_tr = np.array(ptcloud[idx_10][0:3])
                pt_bl = np.array(ptcloud[idx_01][0:3])
                pt_br = np.array(ptcloud[idx_11][0:3])

                ## compute pair-wise distances and exclude triangles who have edges which are too long
                d_tltr = np.linalg.norm(pt_tl-pt_tr)
                d_trbr = np.linalg.norm(pt_tr-pt_br)
                d_brbl = np.linalg.norm(pt_br-pt_bl)
                d_bltl = np.linalg.norm(pt_bl-pt_tl)
                d_trbl = np.linalg.norm(pt_tr-pt_bl)

                if (d_tltr < tri_dist_thres) and (d_trbr < tri_dist_thres) and (d_brbl < tri_dist_thres) and (d_bltl < tri_dist_thres) and (d_trbl < tri_dist_thres):
                    triangles.append([idx_00+1, idx_10+1, idx_01+1]) ## triangle 1
                    triangles.append([idx_10+1, idx_11+1, idx_01+1]) ## triangle 2

            n_triangles_processed += 2
            print_text_progress_bar(n_triangles_processed/(img_h*img_w*2), bar_name='Triangulating....   ')
    print()

    return np.array(ptcloud), np.array(triangles), np.array(texcoords), n_points_total

def output_textured_mesh(ptcloud, texcoords, triangles, fname_image_texture, fname_obj, strOutputPath):
    ''' Textured mesh is in a standard OBJ file format. '''
    texture_file = open(os.path.join(strOutputPath, fname_image_texture) + ".mtl", "w")
    texture_file.write("newmtl material0\n")
    texture_file.write("Ka 1.000000 1.000000 1.000000\n")
    texture_file.write("Kd 1.000000 1.000000 1.000000\n")
    texture_file.write("Ks 0.000000 0.000000 0.000000\n")
    texture_file.write("Tr 1.000000\n")
    texture_file.write("illum 1\n")
    texture_file.write("Ns 0.000000\n")
    texture_file.write("map_Kd "+fname_image_texture+".png\n")
    texture_file.close()

    print("Writing point cloud to '" + str(strOutputPath + fname_obj + ".obj") + "' ... ", end="", flush=True)
    outputFile = open(os.path.join(strOutputPath, fname_obj) + ".obj", "w")
    outputFile.write("mtllib ./"+fname_image_texture+".mtl\n")

    # output the actual points
    for pt in ptcloud:
        dx, dy, dz = pt[0:3]
        outputFile.write("v %.6f %.6f %.6f\n" %(dx, dy, dz))
        outputFile.write("vn 0.0 0.0 0.0\n")

    for vt in texcoords:
        tex_u, tex_v = vt
        outputFile.write("vt %.6f %.6f\n" %(tex_u, tex_v))

    outputFile.write("usemtl material0\n")
    for t in triangles:
        idx_1, idx_2, idx_3 = t
        outputFile.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" %(idx_1, idx_1, idx_1, idx_2, idx_2, idx_2, idx_3, idx_3, idx_3))

    outputFile.close()

    print("Finished!")

