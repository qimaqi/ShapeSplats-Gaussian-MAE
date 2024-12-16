import os
import time

# from joblib import Parallel, delayed
import argparse
import trimesh
import json
import zipfile
import numpy as np
from io import BytesIO
from PIL import Image


def center_and_scale_mesh(mesh):
    # Calculate the centroid of the mesh
    centroid = mesh.centroid
    # Translate the mesh to the origin
    mesh.vertices -= centroid

    # Calculate the axis-aligned bounding box (AABB) extents
    extents = mesh.bounding_box.extents
    max_extent = max(extents) * 2

    # Scale the mesh to fit within the range [-1, 1]
    mesh.vertices /= max_extent / 2

    return mesh


def transform_mesh_axes(mesh):
    # Define the transformation matrix
    # This matrix maps:
    # z to -x
    # x to y
    # y to -z

    # z to y
    # x to -z
    # y tp x
    transformation_matrix = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    mesh.apply_transform(transformation_matrix)
    return mesh


parser = argparse.ArgumentParser()
parser.add_argument("--model_root_dir", type=str, default="./ModelNet40/")
parser.add_argument("--render_root_dir", type=str, default="./ModelNet40/render/")
parser.add_argument("--file_dict_path", type=str, default="./modelnet_data_dict.json")
parser.add_argument(
    "--start_idx", type=int, default=0, help="start scene you want to train."
)
parser.add_argument("--end_idx", type=int, default=10, help="end scene you want to end")
parser.add_argument(
    "--blender_location",
    type=str,
    default="/cluster/work/cvl/qimaqi/3dv_gaussian/blender_install/blender-3.6.13-linux-x64/blender",
)
parser.add_argument("--num_thread", type=int, default=10, help="1/3 of the CPU number")
parser.add_argument("--debug", type=bool, default=False)
FLAGS = parser.parse_args()

model_root_dir = FLAGS.model_root_dir
render_root_dir = FLAGS.render_root_dir

# cat_ids = {
#         "watercraft": "04530566",
#         "rifle": "04090263",
#         "display": "03211117",
#         "lamp": "03636649",
#         "speaker": "03691459",
#         "cabinet": "02933112",
#         "chair": "03001627",
#         "bench": "02828884",
#         "car": "02958343",
#         "airplane": "02691156",
#         "sofa": "04256520",
#         "table": "04379243",
#         "phone": "04401088"
#     }


def gen_obj(model_root_dir, cat_id, split_i, obj_id):
    t_start = time.time()
    print("Start %s %s" % (cat_id, obj_id))

    objpath = os.path.join(model_root_dir, cat_id, split_i, obj_id)  # for v1

    objname = obj_id.split(".")[0]
    obj_save_dir = os.path.join(render_root_dir, cat_id, split_i, objname)
    os.makedirs(obj_save_dir, exist_ok=True)
    print("objpath", objpath)
    print("obj_save_dir", obj_save_dir)
    # "There is no item named '000.png' in the archive” error
    run_flag = True
    if os.path.exists(os.path.join(obj_save_dir, "image.zip")):
        # check image.zip image number
        try:
            with zipfile.ZipFile(
                os.path.join(obj_save_dir, "image.zip"), "r"
            ) as zip_ref:
                zip_contents = zip_ref.namelist()
                if len(zip_contents) == 72:
                    print("Exist!!!, skip %s %s" % (cat_id, obj_id))
                    # load one image check its rgb value
                    image_data = zip_ref.read(zip_contents[0])
                    image_file = BytesIO(image_data)
                    image = Image.open(image_file)
                    image_np = np.array(image)
                    image_np_sum = np.sum(image_np)
                    if image_np_sum == 0:
                        run_flag = True
                        print("image_np_sum", image_np_sum)
                        print("===================================")
                    else:
                        run_flag = False
                else:
                    print(
                        "missing render images", len(zip_contents), cat_id, obj_id
                    )  # 41f9be4d80af2c709c12d6260da9ac2b
                    run_flag = True
        except Exception as e:
            print("error in reading zip file", e)
            run_flag = True

    elif not os.path.exists(objpath):
        print("Non-Exist object model!!!, skip %s %s" % (cat_id, obj_id))
        run_flag = False

    if run_flag:
        print("Start %s %s" % (cat_id, obj_id))
        mesh = trimesh.load(objpath, force="mesh")
        # unlike shapenet, we need to renormalize the object
        centered_and_scaled_mesh = center_and_scale_mesh(mesh)
        centered_and_scaled_mesh = transform_mesh_axes(centered_and_scaled_mesh)

        centered_and_scaled_mesh.export(os.path.join(obj_save_dir, "point_cloud.obj"))
        objpath_debug = os.path.join(obj_save_dir, "point_cloud.obj")
        # print("load mesh")
        # print out range of mesh
        vertices = centered_and_scaled_mesh.vertices
        box_json_path = os.path.join(obj_save_dir, "box.json")

        # render to 2D
        if FLAGS.debug:
            os.system(
                FLAGS.blender_location
                + " --background --python render_blender_modelnet.py -- --views %d --obj_save_dir %s  %s"
                % (72, obj_save_dir, objpath_debug)
            )
        else:
            try:
                os.system(
                    FLAGS.blender_location
                    + "  --background --python render_blender_modelnet.py -- --views %d --obj_save_dir %s %s > /dev/null 2>&1"
                    % (72, obj_save_dir, objpath_debug)
                )
            except Exception as error:
                print("failure rendering", objpath, "due to error", error)
                print("===================================")
                pass

        print("Finished %s %s" % (cat_id, obj_id), time.time() - t_start)


#

with open(FLAGS.file_dict_path, "r") as f:
    data_dict = json.load(f)

model_root_dir_lst = []
cat_id_lst = []
split_lst_all = []
obj_id_lst = []
for cat_id in data_dict.keys():
    lst = []
    split_lst = []
    for split_i in data_dict[cat_id]:
        for obj_id in data_dict[cat_id][split_i]:
            lst.append(obj_id)
            split_lst.append(split_i)

    model_root_dir_i = [model_root_dir for i in range(len(lst))]
    cat_id_lst_i = [cat_id for i in range(len(lst))]
    model_root_dir_lst.extend(model_root_dir_i)
    cat_id_lst.extend(cat_id_lst_i)
    split_lst_all.extend(split_lst)
    obj_id_lst.extend(lst)

if FLAGS.end_idx > len(obj_id_lst) or FLAGS.end_idx == -1:
    FLAGS.end_idx = len(obj_id_lst)

print("total length of obj_total_path", len(obj_id_lst))
model_root_dir_lst = model_root_dir_lst[FLAGS.start_idx : FLAGS.end_idx]
cat_id_lst = cat_id_lst[FLAGS.start_idx : FLAGS.end_idx]
split_lst_all = split_lst_all[FLAGS.start_idx : FLAGS.end_idx]
obj_id_lst = obj_id_lst[FLAGS.start_idx : FLAGS.end_idx]

for model_root_dir, cat_id, split_i, obj_id in zip(
    model_root_dir_lst, cat_id_lst, split_lst_all, obj_id_lst
):
    gen_obj(model_root_dir, cat_id, split_i, obj_id)
