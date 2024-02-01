# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 17:31:42 2022

@author: why
"""

import open3d
import torch
import matplotlib
import numpy as np
import os

box_colormap = [
    [0.5, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


def translate_boxes_to_open3d_instance(gt_boxes):
    """
     4-------- 6
     /| /|
    5 -------- 3 .
    | || |
    . 7 -------- 1
    |/ |/
    2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
         line_set.paint_uniform_color(color)
        else:
          line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
    # corners = box3d.get_box_points()
    # vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis

def main():
    txt_folder = "eval/kitti/object/pred_velo"
    bin_folder = "data"
    if not os.path.exists(txt_folder):
        return "result txt_folder is not exists!"
    bin_lists = os.listdir(bin_folder)
    for file in bin_lists:
        (filename,extension) = os.path.splitext(file)
        result_path = os.path.join(txt_folder,filename) + ".txt"
        bin_path = os.path.join(bin_folder,file)
        if not os.path.exists(result_path):
            print(result_path,"file is not exists!")
            continue
        if not os.path.exists(bin_path):
            print(bin_path,"file is not exists!")
            continue                
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        result_boxes = np.loadtxt(result_path)
        if len(result_boxes) < 1:
            draw_scenes(points=points)
            continue
        if(len(result_boxes.shape)< 2):
            result_boxes = result_boxes.reshape(1,-1)
        ref_boxes = result_boxes[:,:7].astype(float)
        ref_labels = result_boxes[:,7].astype(int)
        ref_scores = result_boxes[:,8].astype(float)
        print("ref_labels:  ",ref_labels)
        print("ref_scores: ",ref_scores)
        draw_scenes(points=points,ref_boxes=ref_boxes,ref_scores=ref_scores, ref_labels=ref_labels)


if __name__ == '__main__':
    main()
