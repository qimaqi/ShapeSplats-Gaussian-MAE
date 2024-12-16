# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# blender --background --python mytest.py -- --views 10 /path/to/my.obj
#

import shutil
from math import radians
import bpy
import numpy as np
import argparse
import sys
import os
import json
import copy

# import trimesh

parser = argparse.ArgumentParser(
    description="Renders given obj file by rotation a camera around it."
)
parser.add_argument(
    "--views", type=int, default=1, help="number of views to be rendered"
)
parser.add_argument("obj", type=str, help="Path to the obj file to be rendered.")
parser.add_argument(
    "--scale",
    type=float,
    default=1,
    help="Scaling factor applied to model. Depends on size of mesh.",
)
parser.add_argument(
    "--remove_doubles",
    type=bool,
    default=False,
    help="Remove double vertices to improve mesh quality.",
)
parser.add_argument(
    "--edge_split", type=bool, default=False, help="Adds edge split filter."
)
parser.add_argument(
    "--depth_scale",
    type=float,
    default=1.4,
    help="Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.",
)
parser.add_argument(
    "--color_depth",
    type=str,
    default="8",
    help="Number of bit per channel used for output. Either 8 or 16.",
)
parser.add_argument(
    "--format",
    type=str,
    default="PNG",
    help="Format of files generated. Either PNG or OPEN_EXR",
)
parser.add_argument("--obj_save_dir", type=str)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)


print("start rendering")


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


# Set up rendering of depth map using nodes
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
bpy.context.view_layer.use_pass_normal = True
bpy.context.view_layer.use_pass_combined = True
bpy.context.view_layer.use_pass_z = True
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.context.scene.render.image_settings.color_depth = str(8)

render_layers_node = tree.nodes.new("CompositorNodeRLayers")

# Setup for output of depth
# Link nodes
links = tree.links
# clear default nodes
# for n in tree.nodes:
#     tree.nodes.remove(n)

depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
depth_file_output.base_path = args.obj_save_dir
depth_file_output.label = "Depth Output"
depth_file_output.name = "Depth Output"
# Remap as other types can not represent the full range of depth.
map = tree.nodes.new(type="CompositorNodeMapRange")
# Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
map.inputs["From Min"].default_value = 0
map.inputs["From Max"].default_value = 8
map.inputs["To Min"].default_value = 1
map.inputs["To Max"].default_value = 0
links.new(render_layers_node.outputs["Depth"], map.inputs[0])
links.new(map.outputs[0], depth_file_output.inputs[0])

#  Setup for output of normals
normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
normal_file_output.base_path = args.obj_save_dir
normal_file_output.label = "Normal Output"
normal_file_output.name = "Normal Output"
links.new(render_layers_node.outputs["Normal"], normal_file_output.inputs[0])

# Import OBJ file
bpy.ops.object.select_all(action="DESELECT")
bpy.ops.object.select_by_type(type="MESH")
bpy.ops.object.delete()
imported_obj = bpy.ops.import_scene.obj(filepath=args.obj)
# Assumes imported obj has one main object
obj_object = bpy.context.selected_objects[0]
# Reset object's location
obj_object.location = (0, 0, 0)
bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
# Scale, remove doubles, add edge split if specified
if args.scale != 1:
    obj_object.scale = (args.scale, args.scale, args.scale)
if args.remove_doubles:
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.remove_doubles()
    bpy.ops.object.mode_set(mode="OBJECT")
if args.edge_split:
    modifier = obj_object.modifiers.new(name="EdgeSplit", type="EDGE_SPLIT")
    modifier.split_angle = 1.32645
    bpy.ops.object.modifier_apply(modifier="EdgeSplit")

# Render
# Background
# bpy.context.scene.render.dither_intensity = 0.0
# bpy.context.scene.render.film_transparent = True

# Setup lights and camera
light_data = bpy.data.lights.new(name="Light", type="POINT")
light_data.energy = 1
light_object = bpy.data.objects.new(name="Light", object_data=light_data)
bpy.context.collection.objects.link(light_object)
light_object.location = (3, 3, 5)


# Render settings
def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


def camera_info(param):
    "params: [theta, phi, rho, x, y, z, f]"
    theta = np.deg2rad(param[0])
    phi = np.deg2rad(param[1])
    # print(param[0],param[1], theta, phi, param[6])

    camY = param[3] * np.sin(phi) * param[6]
    temp = param[3] * np.cos(phi) * param[6]
    camX = temp * np.cos(theta)
    camZ = temp * np.sin(theta)
    cam_pos = np.array([camX, camY, camZ])

    axisZ = cam_pos.copy()
    axisY = np.array([0, 1, 0])
    axisX = np.cross(axisY, axisZ)
    # axisY = np.cross(axisZ, axisX)

    # cam_mat = np.array([unit(axisX), unit(axisY), unit(axisZ)])
    print("cam axis", camX, camY, camZ)
    return camX, -camZ, camY


scene = bpy.context.scene
scene.render.resolution_x = 400
scene.render.resolution_y = 400
scene.render.resolution_percentage = 100

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# GPU
# import bpy
# bpy.context.scene.render.engine = 'CYCLES'
# # Enable Cycles add-on if not already enabled
# bpy.ops.preferences.addon_enable(module="cycles")

# # Set the device_type
# cycles_preferences = bpy.context.preferences.addons['cycles'].preferences
# cycles_preferences.compute_device_type = 'CUDA'
# bpy.context.scene.cycles.device = 'GPU'


# Data to store in JSON file
out_data = {
    "camera_angle_x": bpy.data.objects["Camera"].data.angle_x,
}

# scene.render.alpha_mode = 'TRANSPARENT'
cam = scene.objects["Camera"]
cam.location = (0, -1.0, 1.0)
cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = "PNG"  # set output format to .png


stepsize = 360.0 / args.views
# to make the output reproduce, we fix seed and generate vertical difference at beginning
CIRCLE_FIXED_START = (0, 0, 0)
CIRCLE_FIXED_END = (0.7, 0, 0)
# random vertical
np.random.seed(42)
# vertical_list = np.random.rand(args.views) *  np.pi - np.pi / 4 # upper and down views
vertical_list = np.random.rand(args.views) * np.pi / 4  # most upper views

vertical_diff = CIRCLE_FIXED_END[0] - CIRCLE_FIXED_START[0]
# print("vertical_list", vertical_list)

rotation_mode = "XYZ"

obj_save_dir = args.obj_save_dir

# TODO easy and hard

out_data["frames"] = []
b_empty.rotation_euler = CIRCLE_FIXED_START
b_empty.rotation_euler[0] = vertical_list[0]

# for output_node in [tree.nodes['Depth Output'], tree.nodes['Normal Output']]:
#     output_node.base_path = '/'


current_rot_value = 0
for i in range(args.views):
    # current_rot_value += stepsize
    # counter = 0
    # # while True: #
    # counter+=1
    # angle_rand = np.random.rand(3)
    # y_rot = current_rot_value + angle_rand[0] * 10 - 5
    # x_rot = 20 + angle_rand[1] * 10
    # dist = 0.65 + angle_rand[2] * 0.35

    # param = [y_rot, x_rot, 0, dist, 35, 32, 1.75]
    # camX, camY, camZ = camera_info(param)
    # cam.location = (camX, camY, camZ)

    scene.render.filepath = obj_save_dir + "/image/" + str(i).zfill(3)

    tree.nodes["Depth Output"].file_slots[0].path = "/depth/" + str(i).zfill(3)
    tree.nodes["Normal Output"].file_slots[0].path = "/normal/" + str(i).zfill(3)

    bpy.ops.render.render(write_still=True)  # render still

    frame_data = {
        "file_path": "image/" + str(i).zfill(3),
        "rotation": radians(stepsize),
        "transform_matrix": listify_matrix(cam.matrix_world),
    }
    out_data["frames"].append(frame_data)

    if i == args.views - 1:
        break
    b_empty.rotation_euler[0] = vertical_list[i + 1]
    # CIRCLE_FIXED_START[0] + (np.cos(radians(stepsize*i))+1)/2 * vertical_diff
    # vertical_list[i]
    # CIRCLE_FIXED_START[0] + (np.cos(radians(stepsize*i))+1)/2 * vertical_diff
    b_empty.rotation_euler[2] += radians(stepsize)


with open(obj_save_dir + "/" + "transforms_train.json", "w") as out_file:
    json.dump(out_data, out_file, indent=4)


test_json = copy.deepcopy(out_data)
test_json["frames"] = test_json["frames"][-4:]

with open(os.path.join(obj_save_dir, "transforms_test.json"), "w") as f:
    json.dump(test_json, f, indent=4)

with open(os.path.join(obj_save_dir, "transforms_val.json"), "w") as f:
    json.dump(test_json, f, indent=4)


# zip image file, depth files and normals
print("zip image file, depth files and normals")
shutil.make_archive(
    os.path.join(obj_save_dir, "image"), "zip", os.path.join(obj_save_dir, "image")
)
shutil.make_archive(
    os.path.join(obj_save_dir, "depth"), "zip", os.path.join(obj_save_dir, "depth")
)
shutil.make_archive(
    os.path.join(obj_save_dir, "normal"), "zip", os.path.join(obj_save_dir, "normal")
)

shutil.rmtree(os.path.join(obj_save_dir, "image"))
shutil.rmtree(os.path.join(obj_save_dir, "depth"))
shutil.rmtree(os.path.join(obj_save_dir, "normal"))
# print("vertical_list", vertical_list)
