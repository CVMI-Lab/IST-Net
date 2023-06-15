# introduced from fs-net
import numpy as np
import cv2
import torch
import math


def get_rotation(x_, y_, z_):
    # print(math.cos(math.pi/2))
    x = float(x_ / 180) * math.pi
    y = float(y_ / 180) * math.pi
    z = float(z_ / 180) * math.pi
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(x), -math.sin(x)],
                    [0, math.sin(x), math.cos(x)]])

    R_y = np.array([[math.cos(y), 0, math.sin(y)],
                    [0, 1, 0],
                    [-math.sin(y), 0, math.cos(y)]])

    R_z = np.array([[math.cos(z), -math.sin(z), 0],
                    [math.sin(z), math.cos(z), 0],
                    [0, 0, 1]])
    return np.dot(R_z, np.dot(R_y, R_x)).astype(np.float32)

def get_rotation_torch(x_, y_, z_):
    x = (x_ / 180) * math.pi
    y = (y_ / 180) * math.pi
    z = (z_ / 180) * math.pi
    R_x = torch.tensor([[1, 0, 0],
                    [0, math.cos(x), -math.sin(x)],
                    [0, math.sin(x), math.cos(x)]], device=x_.device)

    R_y = torch.tensor([[math.cos(y), 0, math.sin(y)],
                    [0, 1, 0],
                    [-math.sin(y), 0, math.cos(y)]], device=y_.device)

    R_z = torch.tensor([[math.cos(z), -math.sin(z), 0],
                    [math.sin(z), math.cos(z), 0],
                    [0, 0, 1]], device=z_.device)
    return torch.mm(R_z, torch.mm(R_y, R_x))

# point cloud based data augmentation
# augment based on bounding box
def defor_3D_bb(pc, R, t, s, nocs, model, sym=None, aug_bb=None):
    # pc  n x 3, here s must  be the original s
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    if sym[0] == 1:  # y axis symmetry
        ex = aug_bb[0]
        ey = aug_bb[1]
        ez = aug_bb[2]

        exz = (ex + ez) / 2
        pc_reproj[:, (0, 2)] = pc_reproj[:, (0, 2)] * exz
        pc_reproj[:, 1] = pc_reproj[:, 1] * ey
        nocs_scale_aug = torch.norm(torch.tensor([s[0] * exz, s[1] * ey, s[2] * exz])) / torch.norm(s)
        s[0] = s[0] * exz
        s[1] = s[1] * ey
        s[2] = s[2] * exz
        nocs[:, 0] = nocs[:, 0] * exz / nocs_scale_aug
        nocs[:, 1] = nocs[:, 1] * ey / nocs_scale_aug
        nocs[:, 2] = nocs[:, 2] * exz / nocs_scale_aug
        model[:, 0] = model[:, 0] * exz / nocs_scale_aug
        model[:, 1] = model[:, 1] * ey / nocs_scale_aug
        model[:, 2] = model[:, 2] * exz / nocs_scale_aug
        pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
        pc_new = pc_new.T
        # aug_param = torch.tensor([exz, ey, exz])
    else:
        ex = aug_bb[0]
        ey = aug_bb[1]
        ez = aug_bb[2]
        nocs_scale_aug = torch.norm(torch.tensor([s[0] * ex, s[1] * ey, s[2] * ez])) / torch.norm(s)
        pc_reproj[:, 0] = pc_reproj[:, 0] * ex
        pc_reproj[:, 1] = pc_reproj[:, 1] * ey
        pc_reproj[:, 2] = pc_reproj[:, 2] * ez
        s[0] = s[0] * ex
        s[1] = s[1] * ey
        s[2] = s[2] * ez
        nocs[:, 0] = nocs[:, 0] * ex / nocs_scale_aug
        nocs[:, 1] = nocs[:, 1] * ey / nocs_scale_aug
        nocs[:, 2] = nocs[:, 2] * ez / nocs_scale_aug
        model[:, 0] = model[:, 0] * ex / nocs_scale_aug
        model[:, 1] = model[:, 1] * ey / nocs_scale_aug
        model[:, 2] = model[:, 2] * ez / nocs_scale_aug
        pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
        pc_new = pc_new.T
    # print("s", s)
    # assert False
    return pc_new, s, nocs, model

# point cloud based data augmentation
# random rotation and translation
def defor_3D_rt(pc, R, t, aug_rt_t, aug_rt_r):
    #  add_t
    dx = aug_rt_t[0]
    dy = aug_rt_t[1]
    dz = aug_rt_t[2]

    pc[:, 0] = pc[:, 0] + dx
    pc[:, 1] = pc[:, 1] + dy
    pc[:, 2] = pc[:, 2] + dz
    t[0] = t[0] + dx
    t[1] = t[1] + dy
    t[2] = t[2] + dz

    # add r
    '''
    Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
    Rm_tensor = torch.tensor(Rm, device=pc.device)
    pc_new = torch.mm(Rm_tensor, pc.T).T
    pc = pc_new
    R_new = torch.mm(Rm_tensor, R)
    R = R_new
    '''
    '''
    x_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    y_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    z_rot = torch.rand(1, dtype=torch.float32, device=pc.device) * 2 * a - a
    Rm = get_rotation_torch(x_rot, y_rot, z_rot)
    '''
    Rm = aug_rt_r
    pc_new = torch.mm(Rm, pc.T).T
    pc = pc_new
    R_new = torch.mm(Rm, R)
    R = R_new
    T_new = torch.mm(Rm, t.view(3, 1))
    t = T_new

    return pc, R, t

def defor_3D_bc(pc, R, t, s, model_point, nocs_scale, nocs):
    # resize box cage along y axis, the size s is modified
    ey_up = torch.rand(1, device=pc.device) * (1.2 - 0.8) + 0.8
    ey_down = torch.rand(1,  device=pc.device) * (1.2 - 0.8) + 0.8
    # for each point, resize its x and z linealy
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    per_point_resize = (pc_reproj[:, 1] + s[1] / 2) / s[1] * (ey_up - ey_down) + ey_down
    pc_reproj[:, 0] = pc_reproj[:, 0] * per_point_resize
    pc_reproj[:, 2] = pc_reproj[:, 2] * per_point_resize
    pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
    pc_new = pc_new.T

    norm_s = s / torch.norm(s)
    model_point_resize =  (model_point[:, 1] + norm_s[1] / 2) / norm_s[1] * (ey_up - ey_down) + ey_down
    model_point[:, 0] = model_point[:, 0] * model_point_resize
    model_point[:, 2] = model_point[:, 2] * model_point_resize

    lx = 2 * max(max(model_point[:, 0]), -min(model_point[:, 0]))
    ly = max(model_point[:, 1]) - min(model_point[:, 1])
    lz = max(model_point[:, 2]) - min(model_point[:, 2])

    lx_t = lx * torch.norm(s)
    ly_t = ly * torch.norm(s)
    lz_t = lz * torch.norm(s)
    size_new = torch.tensor([lx_t, ly_t, lz_t], device=pc.device)

    nocs_scale_aug = torch.norm(torch.tensor([lx, ly, lz]))
    model_point = model_point / nocs_scale_aug

    nocs_resize = (nocs[:, 1] + norm_s[1] / 2) / norm_s[1] * (ey_up - ey_down) + ey_down
    nocs[:, 0] = nocs[:, 0] * nocs_resize
    nocs[:, 2] = nocs[:, 2] * nocs_resize
    nocs = nocs / nocs_scale_aug

    return pc_new, size_new, model_point, nocs

def defor_3D_pc(pc, r):
    points_defor = torch.randn(pc.shape).to(pc.device)
    pc = pc + points_defor * r
    return pc

# point cloud based data augmentation
# augment based on bounding box
def deform_non_linear(pc, R, t, s, nocs, model_point, axis=0):
    # pc  n x 3, here s must  be the original s
    assert axis in [0, 1]
    r_max = torch.rand(1, device=pc.device) * 0.2 + 1.1
    r_min = -torch.rand(1, device=pc.device) * 0.2 + 0.9
    # for each point, resize its x and z
    pc_reproj = torch.mm(R.T, (pc - t.view(1, 3)).T).T  # nn x 3
    per_point_resize = r_min + 4 * (pc_reproj[:, axis] * pc_reproj[:, axis]) / (s[axis] ** 2) * (r_max - r_min)
    pc_reproj[:, axis] = pc_reproj[:, axis] * per_point_resize
    pc_new = torch.mm(R, pc_reproj.T) + t.view(3, 1)
    pc_new = pc_new.T

    norm_s = s / torch.norm(s)
    model_point_resize = r_min + 4 * (model_point[:, axis] * model_point[:, axis]) / (norm_s[axis] ** 2) * (r_max - r_min)
    model_point[:, axis] = model_point[:, axis] * model_point_resize

    lx = 2 * max(max(model_point[:, 0]), -min(model_point[:, 0]))
    ly = max(model_point[:, 1]) - min(model_point[:, 1])
    lz = max(model_point[:, 2]) - min(model_point[:, 2])

    lx_t = lx * torch.norm(s)
    ly_t = ly * torch.norm(s)
    lz_t = lz * torch.norm(s)
    size_new = torch.tensor([lx_t, ly_t, lz_t], device=pc.device)

    nocs_scale_aug = torch.norm(torch.tensor([lx, ly, lz]))
    model_point = model_point / nocs_scale_aug

    nocs_resize = r_min + 4 * (nocs[:, axis] * nocs[:, axis]) / (norm_s[axis] ** 2) * (r_max - r_min)
    nocs[:, axis] = nocs[:, axis] * nocs_resize
    nocs = nocs / nocs_scale_aug
    return pc_new, size_new, model_point, nocs

def data_augment(args, PC, gt_R, gt_t, gt_s, sym, aug_bb, aug_rt_t, aug_rt_r, \
                 model_point, nocs_scale, PC_nocs, obj_id):

    ## bounding box augmentation
    prop_bb = torch.rand(1) # 产生随机概率
    # print("args.aug_bb_pro:", args.aug_bb_pro, 
    #       "aug_rt_pro:", args.aug_rt_pro, 
    #       "aug_bc_pro:", args.aug_bc_pro,
    #       "aug_pc_pro:", args.aug_pc_pro,
    #       "aug_nl_pro:", args.aug_nl_pro)
    
    if prop_bb < args.aug_bb_pro: # 概率大于aug的概率则进行aug
        PC_new, gt_s_new, nocs_new, model_new = defor_3D_bb(PC, gt_R,
                                        gt_t, gt_s, PC_nocs, model_point,
                                        sym=sym, aug_bb=aug_bb)
        PC = PC_new
        gt_s = gt_s_new
        PC_nocs = nocs_new
        model_point = model_new

    ## rt aug
    prop_rt = torch.rand(1)
    if prop_rt < args.aug_rt_pro:
        PC_new, gt_R_new, gt_t_new = defor_3D_rt(PC, gt_R, gt_t, aug_rt_t, aug_rt_r)
        PC = PC_new
        gt_R = gt_R_new
        gt_t = gt_t_new.view(-1)
    

    prop_bc = torch.rand(1)
    # only do bc for mug and bowl
    if prop_bc < args.aug_bc_pro and (obj_id == 5 or obj_id == 1):
        PC_new, gt_s_new, model_point_new, nocs_new = defor_3D_bc(PC, gt_R, gt_t, gt_s,
                                        model_point, nocs_scale, PC_nocs)
        PC = PC_new
        gt_s = gt_s_new
        model_point= model_point_new
        PC_nocs = nocs_new
    
    # add noise
    prop_pc = torch.rand(1)
    if prop_pc < args.aug_pc_pro:
        PC_new = defor_3D_pc(PC, args.aug_pc_r)
        PC = PC_new

    # only do bc for mug and bowl
    prop_nl = torch.rand(1)
    if prop_nl < args.aug_nl_pro and (obj_id in [0,1,2,3,5]):
        if obj_id in [0,1,3,5]:
            sel_axis = 1
        elif obj_id in [2]:
            sel_axis = 0
        else:
            sel_axis = None

        PC_new, gt_s_new, model_point_new, nocs_new = deform_non_linear(PC, gt_R, gt_t, gt_s,
                                                                    PC_nocs, model_point, sel_axis)

        PC = PC_new
        gt_s = gt_s_new
        model_point = model_point_new
        PC_nocs = nocs_new
    
    return PC, gt_R, gt_t, gt_s, model_point, PC_nocs

def generate_aug_parameters(s_x=(0.8, 1.2), s_y=(0.8, 1.2), s_z=(0.8, 1.2)):
    # for bb aug
    ex, ey, ez = torch.rand(3)
    ex = ex * (s_x[1] - s_x[0]) + s_x[0] # 随机拉伸 bb
    ey = ey * (s_y[1] - s_y[0]) + s_y[0]
    ez = ez * (s_z[1] - s_z[0]) + s_z[0]
    return torch.tensor([ex, ey, ez]).cuda()

def data_shape_augment_batch(args, PC, gt_R, gt_t, gt_s, sym, \
                 model_point, nocs_scale, PC_nocs, return_aug_param=False):
    bs = PC.shape[0] # batch_size
    ## bounding box augmentation
    aug_param = torch.ones_like(gt_s)
    for i in range(bs):
        prop_bb = torch.rand(1) # 产生随机概率
        # print("args.aug_bb_pro:", args.aug_bb_pro, 
        #       "aug_rt_pro:", args.aug_rt_pro, 
        #       "aug_bc_pro:", args.aug_bc_pro,
        #       "aug_pc_pro:", args.aug_pc_pro,
        #       "aug_nl_pro:", args.aug_nl_pro)
        if prop_bb < args.aug_bb_pro: # 概率大于aug的概率则进行aug
            # if return_aug_param:
            aug_bb = generate_aug_parameters()
            gts_ori = gt_s[i, ...].clone()
            PC_new, gt_s_new, nocs_new, model_new = defor_3D_bb(PC[i, ...], gt_R[i, ...],
                                            gt_t[i, ...], gt_s[i, ...], PC_nocs[i, ...], model_point[i, ...],
                                            sym=sym[i, ...], aug_bb=aug_bb)
            if return_aug_param:
                aug_param[i, ...] = gt_s_new / gts_ori

            # print(PC_nocs[i, ...] == nocs_new)
            # assert False
            PC[i, ...] = PC_new
            gt_s[i, ...] = gt_s_new
            PC_nocs[i, ...] = nocs_new
            model_point[i, ...] = model_new


    if return_aug_param:
        return PC, gt_s, model_point, PC_nocs, aug_param
    else:
        return PC, gt_s, model_point, PC_nocs