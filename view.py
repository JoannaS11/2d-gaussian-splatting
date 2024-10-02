import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render, network_gui
from utils.image_utils import render_net_image
import torch
import os
import numpy as np
import json
import geomdl.exchange
import geomdl
import math
from scipy.spatial.transform import Rotation as R
import utils.general_utils as general_utils

def sigmoid_function(x):
    # x [0,1]: 
    # sigmoid: auf x Werte zw.0 und 1 skaliert, um 0.5 in x Ri verschoben
    y = 1 / (1 + torch.exp(-10 * (x - 0.5)))#min(max(np.tanh(5*(x-0.5))[0], 0), 1)#
    return y

def fct_wrapper(fct, border_low, border_high, a=1, d=0, mirrored = False):
    # fct: input in [0,1] results in values [0,1]
    length = border_high - border_low
    b = 1 / length
    c = border_low
    if mirrored:
        return lambda t_on_line: torch.where((t_on_line <= border_low) | (t_on_line > border_high), 0, a * fct(1 - b * (t_on_line - c)) + d)
    else:
        return lambda t_on_line: torch.where((t_on_line <= border_low) | (t_on_line > border_high), 0, a * fct(b * (t_on_line - c)) + d)

def combine_fcts_two_fct(t_on_line, reverse, *args, div_factor= 2):
    values = 0
    if reverse:
        subtract_factor = -1
    else:
        subtract_factor = 1

    for arg in range(np.shape(args[0])[0]):
        function = args[0][arg]
        k = function(t_on_line)
        values += subtract_factor * k
        if arg % div_factor != 0:
            subtract_factor *= -1
    
    return values

def factor_fct(segments_array, motion_fct, t_on_line, rot_point, vector_to_line, speed, b_spline,scale_point, two_functions_for_one_segment = True):
    list_fcts = []
    scaled_rot_axis_vector= None
    scaled_rot_axis_vector_reversed = None
    scaled_rot_axis_vector_start = None
    scale = None
    scale_reversed = None
    scale_start = None

    if rot_point:
        medial_axis_vector = torch.zeros([2 * segments_array.shape[0], 3]).to(device='cuda')

    if two_functions_for_one_segment:
        for i in range(segments_array.shape[0]):
            mirrored = False
            for k in range(2):
                list_fcts.append(fct_wrapper(motion_fct, segments_array[i, k], segments_array[i, k+1], mirrored=mirrored))
                mirrored = True

            if rot_point:
                vec_spline = torch.Tensor(b_spline.evaluate_list(segments_array[i].tolist()))
                low_vector = vec_spline[1] - vec_spline[0]
                high_vector = vec_spline[1] - vec_spline[2]
                medial_axis_vector[i * 2] = low_vector
                medial_axis_vector[i * 2 + 1] = high_vector

    # for loop, use function to find factor
    factor_reversed = combine_fcts_two_fct(t_on_line, True, list_fcts)
    factor_original = combine_fcts_two_fct(t_on_line, False, list_fcts)

    if rot_point:
        scaled_rot_axis_vector, scaled_rot_axis_vector_reversed, scaled_rot_axis_vector_start = find_angle(list_fcts, t_on_line, vector_to_line, medial_axis_vector, segments_array, speed)

    return factor_original, factor_reversed, scaled_rot_axis_vector, scaled_rot_axis_vector_reversed, scaled_rot_axis_vector_start, scale, scale_reversed, scale_start

def find_angle(list_fcts, t_on_line, vector_to_line, medial_axis_vector, segments, speed):
    # variables to iterate over segments
    x = 0
    # locate space for gradient of sigmoid
    grad_sig = torch.zeros([t_on_line.shape[0]]).to(device='cuda')
    grad_sig_start = torch.zeros([t_on_line.shape[0]]).to(device='cuda')
    subtract_factor = 1
    # locate space for rot_axis vector
    rot_axis_vector = torch.zeros([t_on_line.shape[0], 3],dtype=torch.float32).to(device='cuda')

    quarter_pi = 45#3.1416 / 4
    #k = torch.zero
    for idx, fct in enumerate(list_fcts): # border 1: [segments[i,0], segments[i,1]]; border 2: [segments[i,1], segments[i,2]]
        y = idx % 2
        # update x
        if idx % 2 == 0 and idx != 0:
            x += 1

        border_low = segments[x,y]
        border_high = segments[x,y+1]
        
        if border_low == 0:
            border_low = -0.0001

        

        # calculate gradient of sigmoid for a half segment
        grad_sig += 4 * subtract_factor * torch.where((t_on_line <= border_low) | (t_on_line > border_high), 0, fct(t_on_line) * (1 - fct(t_on_line)))
        z = torch.argwhere((t_on_line > border_low) & (t_on_line <= border_high))
        t_sort = torch.argsort(t_on_line)
        #k = grad_sig[z]
        #o = torch.where((t_on_line <= border_low) | (t_on_line > border_high), 0, fct(t_on_line) * (1 - fct(t_on_line)))
        #print(k)
        t_sort = torch.argsort(t_on_line)
        #t_m = grad_sig[t_sort]


        if subtract_factor > 0:
            grad_sig_start += 4 * torch.where((t_on_line <= border_low) | (t_on_line > border_high), 0, fct(t_on_line) * (1 - fct(t_on_line)))

        if idx % 2 != 0:
            subtract_factor *= -1

        # calculate rotation axis
        vec_to_line_part_arg = torch.argwhere((t_on_line > border_low) & (t_on_line <= border_high))
        vec_to_line_part = torch.zeros(vector_to_line.shape,dtype=torch.float64).to(device='cuda')
        vec_to_line_part[vec_to_line_part_arg] = vector_to_line[vec_to_line_part_arg]

        j = torch.cross( medial_axis_vector[idx].to(torch.float64).reshape([1,3]), vec_to_line_part)
        #print(f"rot axis vector vor norm {j} and vec to line part {vec_to_line_part}")
        rot_axis_vector += torch.nn.functional.normalize(j, dim=1).to(torch.float32)

    angle = grad_sig * quarter_pi * speed
    angle_reversed = angle * (-1)

    t_sort = torch.argsort(t_on_line)
    t_m = grad_sig[t_sort]

    angle_start = grad_sig_start * quarter_pi * speed

    #angle = (5 * torch.where(angle < 0.001, 0,1)).to(device='cuda')
    #angle_reversed = (5 * torch.where(angle_reversed >-0.001, 0,1)).to(device='cuda')
    #angle_start = (5 * torch.where(angle_start <0.001, 0,1)).to(device='cuda')

    scaled_rot_axis_vector = rot_axis_vector * angle.unsqueeze(1)
    scaled_rot_axis_vector_reversed = rot_axis_vector * angle_reversed.unsqueeze(1)
    scaled_rot_axis_vector_start = rot_axis_vector * angle_start.unsqueeze(1)
    #print(f"scaled rot axis vector {scaled_rot_axis_vector} and start {scaled_rot_axis_vector_start}")
    scaled_rot_axis_vector = torch.from_numpy(R.from_rotvec(scaled_rot_axis_vector.cpu().detach().numpy(),degrees=True).as_quat()).to(dtype=torch.float32, device='cuda')
    scaled_rot_axis_vector_reversed = torch.from_numpy(R.from_rotvec(scaled_rot_axis_vector_reversed.cpu().detach().numpy(),degrees=True).as_quat()).to(dtype=torch.float32, device='cuda')
    scaled_rot_axis_vector_start = torch.from_numpy(R.from_rotvec(scaled_rot_axis_vector_start.cpu().detach().numpy(),degrees=True).as_quat()).to(dtype=torch.float32, device='cuda')

    #scaled_rot_axis_vector = change_quat_from_last_to_first_order(scaled_rot_axis_vector)
    #scaled_rot_axis_vector_reversed = change_quat_from_last_to_first_order(scaled_rot_axis_vector_reversed)
    #scaled_rot_axis_vector_start = change_quat_from_last_to_first_order(scaled_rot_axis_vector_start)

    return scaled_rot_axis_vector, scaled_rot_axis_vector_reversed, scaled_rot_axis_vector_start

def matrix_array_to_quaternion(matrix):
    eps = 1e-7
    w_2 = 0.25 * (1 + matrix[:, 0, 0] + + matrix[:, 1, 1] + matrix[:, 2, 2]) # https://www.semanticscholar.org/paper/Animating-rotation-with-quaternion-curves-Shoemake/8033e0edb3b43c4ba3605d70d0de14efbe69c976
    w = torch.where(w_2 > eps, torch.sqrt(w_2), 0)
    x = torch.where(w_2 > eps, (matrix[:, 1, 2] - matrix[:, 2, 1]) / (w * 4), 0)
    y = torch.where(w_2 > eps, (matrix[:, 2, 0] - matrix[:, 0, 2]) / (w * 4), 0)
    z = torch.where(w_2 > eps, (matrix[:, 0, 1] - matrix[:, 1, 0]) / (w * 4), 0)

    #w += torch.where(w_2 <= eps, 0 , 0)
    x_2 = torch.where(w_2 <= eps, - 0.5 * (matrix[:, 1, 1] + matrix[:, 2, 2]), 1)
    x += torch.where((w_2 <= eps) & (x_2 > eps), torch.sqrt(x_2), 0)
    y += torch.where((w_2 <= eps) & (x_2 > eps), matrix[:, 0, 1] / (2 * x), 0)
    z += torch.where((w_2 <= eps) & (x_2 > eps), matrix[:, 0, 2] / (2 * x), 0)

    #x += torch.where((w_2 <= eps) & (x_2 <= eps), 0 , 0)
    y_2 = torch.where((w_2 <= eps) & (x_2 <= eps),0.5 * (1 - matrix[:, 2, 2]), 1)
    y += torch.where((w_2 <= eps) & (x_2 <= eps) & (y_2 > eps), torch.sqrt(y_2), 0)
    z += torch.where((w_2 <= eps) & (x_2 <= eps) & (y_2 > eps), matrix[:, 1, 2] / (2 * y), 0)

    #y += torch.where((w_2 <= eps) & (x_2 <= eps) & (y_2 <= eps), 0 , 0)
    z += torch.where((w_2 <= eps) & (x_2 <= eps) & (y_2 <= eps), 1, 0)
    """if w_2 > eps:
        w = torch.sqrt(w_2)
        w4 = 4 * w
        x = (matrix[:, 1, 2] - matrix[:, 2, 1] / w4)
        y = (matrix[:, 2, 0] - matrix[:, 0, 2] / w4)
        z = (matrix[:, 0, 1] - matrix[:, 1, 0] / w4)
    else:
        w = torch.zeros([matrix.shape[0],1])
        x_2 = - 0.5 * (matrix[:, 1, 1] + matrix[:, 2, 2])
        if x_2 > eps:
            x = torch.sqrt(x_2)
            x2 = 2 * x
            y = matrix[:, 0, 1] / x2
            z = matrix[:, 0, 2] / x2
        else:
            x = torch.zeros([matrix.shape[0],1])
            y_2 = 0.5 * (1 - matrix[:, 2, 2])
            if y_2 > eps:
                y = torch.sqrt(y_2)
                z = matrix[:, 1, 2] / (2 * y)
            else:
                y = torch.zeros([matrix.shape[0],1])
                z = torch.ones([matrix.shape[0],1])"""

    quat = torch.stack([x,y,z,w], axis=1)
    return quat

def change_quat_from_last_to_first_order(quat): # x,y,z,w --> w,x,y,z
    tmp_x = quat[0,:]
    quat[0,:] = quat[3,:]
    quat[3,:] = quat[2,:]
    quat[2,:] = quat[1,:]
    quat[1,:] = tmp_x

    return quat

def change_quat_from_first_to_last_order(quat): # w,x,y,z --> x,y,z,w
    tmp_w = quat[0,:]
    quat[0,:] = quat[1,:]
    quat[1,:] = quat[2,:]
    quat[2,:] = quat[3,:]
    quat[3,:] = tmp_w

    return quat


def get_segments_array(min_distances, div_factor_lower = 0.5, div_factor_higher=0.5):
    segments_array = np.zeros([min_distances.shape[0], 3]) # [lower_border, t_min_distance, higher_border]
    for i in range(min_distances.shape[0]):
        segments_array[i,1] = min_distances[i,0]
        if i ==  0:
            segments_array[i,0] = 0
        else: 
            segments_array[i,0] = min_distances[i,0] - (min_distances[i,0] - min_distances[i-1,0]) * div_factor_lower

        if i == min_distances.shape[0]-1:
            segments_array[i,2] = 1
        else:
            segments_array[i,2] = min_distances[i,0] + (min_distances[i+1,0] - min_distances[i,0]) * div_factor_higher

    return segments_array


def view(dataset, pipe, iteration, moving,rot_point, json_file_path, data_directory, scale_point):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

   

    if moving: 
        speed = 0.01
        contraction_factor = 0.5
        medial_axis_bspline, t_on_line, vector_to_line, min_distances = load_data_from_json(json_file_path, data_directory)
        # calculate segments
        segments_array = torch.from_numpy(get_segments_array(min_distances)).to(device='cuda')

        motion_fct = sigmoid_function
        two_functions_for_one_segment = True

        factor_original, factor_reversed, scaled_rot_axis_vector, scaled_rot_axis_vector_reversed, scaled_rot_axis_vector_start, scale, scale_reversed, scale_start = factor_fct(segments_array, motion_fct, t_on_line, rot_point, vector_to_line, speed, medial_axis_bspline, scale_point, two_functions_for_one_segment)

        reverse = False
        #gaussians._xyz *= o.to(device='cuda')

        idx = 0
        while True:
            with torch.no_grad():
                    if network_gui.conn == None:
                        network_gui.try_connect(dataset.render_items)
                    while network_gui.conn != None: 
                        try:   
                            t_smallest_distance_arg = torch.argwhere(t_on_line == min_distances[idx % 2, 0])

                            original_point = torch.FloatTensor(medial_axis_bspline.evaluate_single(min_distances[idx % 2, 0].item())).to(device='cuda')
                            half_point = (original_point + contraction_factor * vector_to_line[t_smallest_distance_arg[0], :])[0].to(device='cuda')
                            dist_half_contr_point = math.dist(original_point, half_point)#.to(device='cuda')

                            stop_condition = lambda original_points, moving_parts: math.dist(original_points, moving_parts) > dist_half_contr_point

                            if reverse:
                                factor = factor_reversed
                                if rot_point:
                                    rotation_matrix = scaled_rot_axis_vector_reversed
                            else:
                                if idx == 0:
                                    factor = torch.fmax(torch.zeros_like(factor_original), factor_original)
                                    if rot_point:
                                        rotation_matrix = scaled_rot_axis_vector_start
                                else:
                                    factor = factor_original
                                    if rot_point:
                                        rotation_matrix = scaled_rot_axis_vector
                                
                            factor = torch.reshape(factor, [np.shape(factor)[0 ], 1])
                            #t = torch.tensor([0.5, 0.5, 1,0],dtype=torch.float32)
                            #rotation_matrix = t.repeat(gaussians._xyz.shape[0],1)
                            # while loop
                            #q = 0
                            #while q < 1:
                            #    q += 0.01
                                #print(q)
                            while stop_condition(original_point, gaussians._xyz[t_smallest_distance_arg[0]][0]):
                                net_image_bytes = None
                                custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()

                                """if math.dist(original_point, gaussians._xyz[t_smallest_distance_arg[0]][0]) <= 1.1 * dist_half_contr_point:
                                    slow_down_factor = 10 * (math.dist(original_point, gaussians._xyz[t_smallest_distance_arg[0]][0]) / dist_half_contr_point - 1)
                                    gaussians._xyz += 0.4 *(1 + 1.5*slow_down_factor) * speed * factor * vector_to_line
                                elif math.dist(original_point, gaussians._xyz[t_smallest_distance_arg[0]][0]) >= 1.9 * dist_half_contr_point:
                                    slow_down_factor = 10 * (2 - math.dist(original_point, gaussians._xyz[t_smallest_distance_arg[0]][0]) / (dist_half_contr_point))
                                    gaussians._xyz += 0.4 *(1 + 1.5 * slow_down_factor)* speed * factor * vector_to_line
                                else:"""
                                gaussians._xyz += speed * factor * vector_to_line
                                if rot_point:
                                    #m = torch.from_numpy(R.from_quat(gaussians._rotation.cpu().detach().numpy()).as_matrix())
                                    #print(f"rot vec as matrix {m}")
                                    #h = R.from_rotvec(rotation_matrix.cpu().detach().numpy())
                                    #print(f"gauß rot as matrix{h}")
                                    gaussian_rot = change_quat_from_first_to_last_order(gaussians._rotation)
                                    #rot_matrix = change_quat_from_first_to_last_order(rotation_matrix)
                                    l = torch.matmul(torch.from_numpy(R.from_quat(rotation_matrix.cpu().detach().numpy()).as_matrix())[:], torch.from_numpy(R.from_quat(gaussian_rot.cpu().detach().numpy()).as_matrix())[:]).to(device='cuda')
                                    #rot_matrix = change_quat_from_last_to_first_order(rotation_matrix)
                                    #l = torch.matmul(general_utils.build_rotation(rot_matrix), general_utils.build_rotation(gaussians._rotation))

                                    #gaussians._rotation = matrix_array_to_quaternion(l.to(dtype=torch.float32))
                                    #print(f"new rot matrix {l}")
                                    tmp = torch.from_numpy(R.from_matrix(l.cpu().detach().numpy()).as_quat()).to(dtype=torch.float32, device='cuda')
                                    gaussians._rotation = change_quat_from_last_to_first_order(tmp)

                                #print(f"gaus rot innen {gaussians._rotation[60000]}")
                                if custom_cam != None:
                                    render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                                    net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                                metrics_dict = {
                                    "#": gaussians.get_opacity.shape[0]
                                    # Add more metrics as needed
                                }
                                network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                            idx += 1#torch.tensor(1, dtype=torch.long)
                            #print("außerhalb")
                            if reverse:
                                reverse = False
                            else:
                                reverse = True
                            #print(f"gaus rot {gaussians._rotation[62000]}")
                        except Exception as e:
                                    raise e
                                    print('Viewer closed')
                                    exit(0)
    else:
        while True:
            with torch.no_grad():
                if network_gui.conn == None:
                    network_gui.try_connect(dataset.render_items)
                while network_gui.conn != None:
                    try:
                        net_image_bytes = None
                        custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                        if custom_cam != None:
                            render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                            net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                            net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                        metrics_dict = {
                            "#": gaussians.get_opacity.shape[0]
                            # Add more metrics as needed
                        }
                        network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    except Exception as e:
                        raise e
                        print('Viewer closed')
                        exit(0)

def load_data_from_json(json_file_path, data_directory):
    if json_file_path == "empty":
        print("json_file_path is missing")
        exit()
    
    current_dir = data_directory
    with open(json_file_path, "r+") as input_file:
        input_liste = json.load(input_file)

        # extract data from json file
        dir_json = os.path.join(*input_liste["dir"])
        data_path = os.path.join(current_dir, *input_liste["data"])
        medial_axis_bspline_path = os.path.join(
            current_dir, *input_liste["dir"], *input_liste["medial_axis_spline"]
        )
        data_name = input_liste["data"][-1]
        data_name = data_name.replace(".ply", "")
        t_on_line_path = input_liste["t_on_line"][1:]
        local_min_path = input_liste["min_distances_values"][1:]


        medial_axis_bspline = geomdl.exchange.import_json(medial_axis_bspline_path)[0]

        # read motion array file
        motion_arrays = np.load(os.path.join(current_dir, dir_json, *t_on_line_path))
        t_on_line = motion_arrays["t_on_line"]
        vector_to_line = motion_arrays["vector_to_line"]
        vector_to_line_distances = motion_arrays["vector_to_line_distances"]
        min_distances = np.load(os.path.join(current_dir, dir_json, *local_min_path))
        min_distances = min_distances["local_mins"]
        
        return medial_axis_bspline, torch.from_numpy(t_on_line).to(device='cuda'), torch.from_numpy(vector_to_line).to(device='cuda'), min_distances 

if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--iteration', type=int, default=30000)
    parser.add_argument('--moving_bool', type=bool, default=False)
    parser.add_argument('--json_file_path', type= str, default="empty")
    parser.add_argument('--data_directory', type= str, default="/home/yn86eniw/Documents/Bachelorarbeit2")
    parser.add_argument('--rot_point_bool', type=bool, default = False)
    parser.add_argument('--scale_point_bool', type=bool, default = False)
    args = parser.parse_args(sys.argv[1:])
    print("View: " + args.model_path)
    network_gui.init(args.ip, args.port)
    
    view(lp.extract(args), pp.extract(args), args.iteration, args.moving_bool, args.rot_point_bool, args.json_file_path, args.data_directory, args.scale_point_bool)

    print("\nViewing complete.")