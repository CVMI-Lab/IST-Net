import os
import math
import cv2
import glob
import numpy as np
import _pickle as cPickle
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from data_augmentation import data_augment, get_rotation

from data_utils import load_depth, load_composed_depth, get_bbox, fill_missing

from common_utils import write_obj

class TrainingDataset(Dataset):
    def __init__(self, config, data_dir, data_type='real', num_img_per_epoch=-1, use_fill_miss=True, use_composed_img=True, per_obj=''):
        self.config = config
        self.data_dir = data_dir
        self.data_type = data_type
        self.use_shape_aug = config.get("use_shape_aug", False)
        self.num_img_per_epoch = num_img_per_epoch

        self.use_fill_miss = use_fill_miss
        self.use_composed_img = use_composed_img

        self.img_size = self.config.img_size
        self.sample_num = self.config.sample_num

        if data_type == 'syn':
            img_path = 'CAMERA/train_list.txt'
            model_path = 'obj_models/camera_train.pkl'
            self.intrinsics = [577.5, 577.5, 319.5, 239.5]
        elif data_type == 'real_withLabel':
            img_path = 'Real/train_list.txt'
            model_path = 'obj_models/real_train.pkl'
            self.intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
        else:
            assert False, 'wrong data type of {} in data loader !'.format(data_type)

        img_list = [os.path.join(img_path.split('/')[0], line.rstrip('\n'))
                        for line in open(os.path.join(self.data_dir, img_path))]
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.cat_name2id = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}
        self.id2cat_name_CAMERA = {'1': '02876657',
                                   '2': '02880940',
                                   '3': '02942699',
                                   '4': '02946921',
                                   '5': '03642806',
                                   '6': '03797390'}
        if data_type == "syn":
            self.id2cat_name = self.id2cat_name_CAMERA
        else:
            self.id2cat_name = {'1': 'bottle', '2': 'bowl', '3': 'camera', '4': 'can', '5': 'laptop', '6': 'mug'}
        self.per_obj = per_obj
        self.per_obj_id = None
        if self.per_obj in self.cat_names:
            self.per_obj_id = self.cat_name2id[self.per_obj]
            img_list_cache_dir = os.path.join(self.data_dir, 'img_list')
            if not os.path.exists(img_list_cache_dir):
                os.makedirs(img_list_cache_dir)
            img_list_cache_filename = os.path.join(img_list_cache_dir, f'{per_obj}_{data_type}_img_list.txt')
            if os.path.exists(img_list_cache_filename):
                print(f'read image list cache from {img_list_cache_filename}')
                img_list_obj = [line.rstrip('\n') for line in open(os.path.join(data_dir, img_list_cache_filename))]
            else:
                # needs to reorganize img_list
                s_obj_id = self.cat_name2id[self.per_obj]
                img_list_obj = []
                from tqdm import tqdm
                for i in tqdm(range(len(img_list))):
                    gt_path = os.path.join(self.data_dir, img_list[i] + '_label.pkl')
                    try:
                        with open(gt_path, 'rb') as f:
                            gts = cPickle.load(f)
                        id_list = gts['class_ids']
                        if s_obj_id in id_list:
                            img_list_obj.append(img_list[i])
                    except:
                        print(f'WARNING {gt_path} is empty')
                        continue
                with open(img_list_cache_filename, 'w') as f:
                    for img_path in img_list_obj:
                        f.write("%s\n" % img_path)
                print(f'save image list cache to {img_list_cache_filename}')
            img_list = img_list_obj

        self.img_list = img_list
        self.img_index = np.arange(len(self.img_list))


        self.models = {}
        with open(os.path.join(self.data_dir, model_path), 'rb') as f:
            self.models.update(cPickle.load(f))

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.sym_ids = [0, 1, 3]    # 0-indexed
        self.norm_scale = 1000.0    # normalization scale
        self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        print('{} images found.'.format(len(self.img_list)))
        print('{} models loaded.'.format(len(self.models))) 

    def __len__(self):
        if self.num_img_per_epoch == -1:
            return len(self.img_list)
        else:
            return self.num_img_per_epoch

    def reset(self):
        assert self.num_img_per_epoch != -1
        num_img = len(self.img_list)
        if num_img <= self.num_img_per_epoch:
            self.img_index = np.random.choice(num_img, self.num_img_per_epoch)
        else:
            self.img_index = np.random.choice(num_img, self.num_img_per_epoch, replace=False)

    def generate_aug_parameters(self, s_x=(0.8, 1.2), s_y=(0.8, 1.2), s_z=(0.8, 1.2), ax=50, ay=50, az=50, a=15):
        # for bb aug
        ex, ey, ez = np.random.rand(3)
        ex = ex * (s_x[1] - s_x[0]) + s_x[0] # 随机拉伸 bb
        ey = ey * (s_y[1] - s_y[0]) + s_y[0]
        ez = ez * (s_z[1] - s_z[0]) + s_z[0]
        # for R, t aug
        Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a)) # 获得随机的旋转，在每个角度随机旋转-15，15度
        dx = np.random.rand() * 2 * ax - ax # translation aug
        dy = np.random.rand() * 2 * ay - ay # translation aug
        dz = np.random.rand() * 2 * az - az # translation aug
        return np.array([ex, ey, ez], dtype=np.float32), np.array([dx, dy, dz], dtype=np.float32) / 1000.0, Rm

    def get_sym_info(self, c, mug_handle=1):
        #  sym_info  c0 : face classfication  c1, c2, c3:Three view symmetry, correspond to xy, xz, yz respectively
        # c0: 0 no symmetry 1 axis symmetry 2 two reflection planes 3 unimplemented type
        #  Y axis points upwards, x axis pass through the handle, z axis otherwise
        #
        # for specific defination, see sketch_loss
        if c == 'bottle':
            sym = np.array([1, 1, 0, 1], dtype=np.int)
        elif c == 'bowl':
            sym = np.array([1, 1, 0, 1], dtype=np.int)
        elif c == 'camera':
            sym = np.array([0, 0, 0, 0], dtype=np.int)
        elif c == 'can':
            sym = np.array([1, 1, 1, 1], dtype=np.int)
        elif c == 'laptop':
            sym = np.array([0, 1, 0, 0], dtype=np.int)
        elif c == 'mug' and mug_handle == 1:
            sym = np.array([0, 1, 0, 0], dtype=np.int)  # for mug, we currently mark it as no symmetry
        elif c == 'mug' and mug_handle == 0:
            sym = np.array([1, 0, 0, 0], dtype=np.int)
        else:
            sym = np.array([0, 0, 0, 0], dtype=np.int)
        return sym


    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_list[self.img_index[index]])
        if self.data_type == 'syn' and self.use_composed_img:
            # print("img_path:", img_path)
            depth = load_composed_depth(img_path)
        else:
            depth = load_depth(img_path)
        if depth is None:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)
        if self.use_fill_miss:
            depth = fill_missing(depth, self.norm_scale, 1)

        # mask
        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        num_instance = len(gts['instance_ids'])
        assert(len(gts['class_ids'])==len(gts['instance_ids']))
        mask = cv2.imread(img_path + '_mask.png')[:, :, 2] #480*640

        if self.per_obj != '':
            idx = gts['class_ids'].index(self.per_obj_id)
        else:
            idx = np.random.randint(0, num_instance)
        cat_id = gts['class_ids'][idx] - 1 # convert to 0-indexed
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx])
        mask = np.equal(mask, gts['instance_ids'][idx])             

        mask = np.logical_and(mask , depth > 0)

        # choose
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose)<=0:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)
        if len(choose) <= self.sample_num:
            choose_idx = np.random.choice(len(choose), self.sample_num)
        else:
            choose_idx = np.random.choice(len(choose), self.sample_num, replace=False)
        choose = choose[choose_idx]

        # pts
        cam_fx, cam_fy, cam_cx, cam_cy = self.intrinsics
        pts2 = depth.copy() / self.norm_scale
        pts0 = (self.xmap - cam_cx) * pts2 / cam_fx
        pts1 = (self.ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) # 480*640*3
        pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :]
        pts = pts + np.clip(0.001*np.random.randn(pts.shape[0], 3), -0.005, 0.005)

        # rgb
        rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1] #480*640*3
        rgb = rgb[rmin:rmax, cmin:cmax, :]
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        # color jitter
        rgb = self.colorjitter(Image.fromarray(np.uint8(rgb)))
        rgb = self.transform(np.array(rgb))

        # update choose
        crop_w = rmax - rmin
        ratio = self.img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)


        ret_dict = {}
        ret_dict['pts'] = torch.FloatTensor(pts) # N*3
        ret_dict['rgb'] = torch.FloatTensor(rgb)
        ret_dict['choose'] = torch.IntTensor(choose).long()
        ret_dict['category_label'] = torch.IntTensor([cat_id]).long()

        if self.data_type == 'syn' or self.data_type == 'real_withLabel':
            model = self.models[gts['model_list'][idx]].astype(np.float32)
            translation = gts['translations'][idx].astype(np.float32)
            rotation = gts['rotations'][idx].astype(np.float32)
            size = gts['scales'][idx] * gts['sizes'][idx].astype(np.float32)

            if cat_id in self.sym_ids:
                theta_x = rotation[0, 0] + rotation[2, 2]
                theta_y = rotation[0, 2] - rotation[2, 0]
                r_norm = math.sqrt(theta_x**2 + theta_y**2)
                s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                                    [0.0,            1.0,  0.0           ],
                                    [theta_y/r_norm, 0.0,  theta_x/r_norm]])
                rotation = rotation @ s_map
            qo = (pts - translation[np.newaxis, :]) / (np.linalg.norm(size)+1e-8) @ rotation

            sRT = np.identity(4, dtype=np.float32)
            sRT[:3, :3] = gts['scales'][idx] * rotation
            sRT[:3, 3] = translation


            ret_dict['model'] = torch.FloatTensor(model)
            ret_dict['qo'] = torch.FloatTensor(qo)
            ret_dict['translation_label'] = torch.FloatTensor(translation)
            
            ret_dict['rotation_label'] = torch.FloatTensor(rotation)
            ret_dict['size_label'] = torch.FloatTensor(size)
            sym_info = self.get_sym_info(self.id2cat_name[str(cat_id + 1)], mug_handle=1)
            ret_dict['sym_info'] =  torch.IntTensor(sym_info).long()
            # generate augmentation parameters
            if self.use_shape_aug:
                bb_aug, rt_aug_t, rt_aug_R = self.generate_aug_parameters()
                # if cat_id == 5 and self.data_type == 'real_withLabel':
                #     mug_handle = 
                # else:
                #     mug_handle = 1

                # sym_info = self.get_sym_info(self.id2cat_name[str(cat_id + 1)], mug_handle=1)

                aug_bb = torch.as_tensor(bb_aug, dtype=torch.float32).contiguous()
                aug_rt_t = torch.as_tensor(rt_aug_t, dtype=torch.float32).contiguous()
                aug_rt_r = torch.as_tensor(rt_aug_R, dtype=torch.float32).contiguous() 

                # ori_model = ret_dict['model'].clone()
                # ori_nocs = ret_dict['qo'].clone()
                PC_da, gt_R_da, gt_t_da, gt_s_da, model_point, PC_nocs = data_augment(self.config, ret_dict['pts'], ret_dict['rotation_label'],
                                                                                        ret_dict['translation_label'], ret_dict['size_label'], sym_info,
                                                                                        aug_bb, aug_rt_t, aug_rt_r, ret_dict['model'], gts['scales'][idx], ret_dict['qo'],
                                                                                        ret_dict['category_label'])
                
                sRT = np.identity(4, dtype=np.float32)
                sRT[:3, :3] = torch.norm(gt_s_da) * gt_R_da
                sRT[:3, 3] = gt_t_da

                ret_dict['pts'] = PC_da
                ret_dict['rotation_label'] = gt_R_da
                ret_dict['translation_label'] = gt_t_da
                ret_dict['size_label'] = gt_s_da
                ret_dict['model'] = model_point
                ret_dict['qo'] = PC_nocs

        return ret_dict


class TestDataset():
    def __init__(self, config, data_dir):
        self.data_dir = data_dir
        model_path = 'obj_models/real_test.pkl'

        self.img_size = config.img_size
        self.sample_num = config.sample_num
        self.intrinsics = [591.0125, 590.16775, 322.525, 244.11084]

        result_pkl_list = glob.glob(os.path.join(self.data_dir, 'data', 'segmentation_results', 'test_trainedwithMask', 'results_*.pkl'))
        self.result_pkl_list = sorted(result_pkl_list)
        n_image = len(result_pkl_list)
        print('no. of test images: {}\n'.format(n_image))

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.sym_ids = [0, 1, 3]    # 0-indexed
        self.norm_scale = 1000.0    # normalization scale
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        self.class_name_map = {1: 'bottle_',
                               2: 'bowl_',
                               3: 'camera_',
                               4: 'can_',
                               5: 'laptop_',
                               6: 'mug_'}
        self.models = {}
        with open(os.path.join(self.data_dir, 'data', model_path), 'rb') as f:
            self.models.update(cPickle.load(f))

    def __len__(self):
        return len(self.result_pkl_list)

    def __getitem__(self, index):
        path = self.result_pkl_list[index]

        with open(path, 'rb') as f:
            data = cPickle.load(f)

        # assert False
        image_path = os.path.join(self.data_dir, data['image_path'])
        image_path = image_path.replace('/data/real/', '/data/Real/')
        
        pred_data = data
        pred_mask = data['pred_masks']

        num_instance = len(pred_data['pred_class_ids'])
    
        # rgb
        rgb = cv2.imread(image_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1] #480*640*3

        # nocs
        coord = cv2.imread(image_path + '_coord.png')[:, :, :3]
        coord = coord[:, :, (2, 1, 0)]
        coord = np.array(coord, dtype=np.float32) / 255
        coord[:, :, 2] = 1 - coord[:, :, 2]


        # pts
        cam_fx, cam_fy, cam_cx, cam_cy = self.intrinsics
        depth = load_depth(image_path) #480*640
        depth = fill_missing(depth, self.norm_scale, 1)

        xmap = self.xmap
        ymap = self.ymap
        pts2 = depth.copy() / self.norm_scale
        pts0 = (xmap - cam_cx) * pts2 / cam_fx
        pts1 = (ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) # 480*640*3
        
        all_rgb = []
        all_nocs = []
        all_pts = []
        all_models = []
        all_cat_ids = []
        all_choose = []
        flag_instance = torch.zeros(num_instance) == 1


        for j in range(num_instance):
            inst_mask = 255 * pred_mask[:, :, j].astype('uint8')
            rmin, rmax, cmin, cmax = get_bbox(pred_data['pred_bboxes'][j])
            mask = inst_mask > 0
            mask = np.logical_and(mask, depth>0)
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose)>16:
                if len(choose) <= self.sample_num:
                    choose_idx = np.random.choice(len(choose), self.sample_num)
                else:
                    choose_idx = np.random.choice(len(choose), self.sample_num, replace=False)
                choose = choose[choose_idx]
                instance_pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :]

                instance_nocs = coord[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :] - 0.5

                instance_rgb = rgb[rmin:rmax, cmin:cmax, :].copy()
                instance_rgb = cv2.resize(instance_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                instance_rgb = self.transform(np.array(instance_rgb))
                crop_w = rmax - rmin
                ratio = self.img_size / crop_w
                col_idx = choose % crop_w
                row_idx = choose // crop_w
                choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)

                cat_id = pred_data['pred_class_ids'][j] - 1 # convert to 0-indexed

                all_pts.append(torch.FloatTensor(instance_pts))
                all_rgb.append(torch.FloatTensor(instance_rgb))
                all_nocs.append(torch.FloatTensor(instance_nocs))
                all_cat_ids.append(torch.IntTensor([cat_id]).long())
                all_choose.append(torch.IntTensor(choose).long())
                flag_instance[j] = 1

        ret_dict = {}
        ret_dict['pts'] = torch.stack(all_pts) # N*3
        ret_dict['rgb'] = torch.stack(all_rgb)
        ret_dict['ori_img'] = torch.tensor(cv2.imread(image_path + '_color.png')[:, :, :3])
        ret_dict['nocs'] = torch.stack(all_nocs)
        ret_dict['choose'] = torch.stack(all_choose)
        ret_dict['category_label'] = torch.stack(all_cat_ids).squeeze(1)

        ret_dict['gt_class_ids'] = torch.tensor(data['gt_class_ids'])
        ret_dict['gt_bboxes'] = torch.tensor(data['gt_bboxes'])
        ret_dict['gt_RTs'] = torch.tensor(data['gt_RTs'])
        ret_dict['gt_scales'] = torch.tensor(data['gt_scales'])
        ret_dict['gt_handle_visibility'] = torch.tensor(data['gt_handle_visibility'])

        ret_dict['pred_class_ids'] = torch.tensor(pred_data['pred_class_ids'])[flag_instance==1]
        ret_dict['pred_bboxes'] = torch.tensor(pred_data['pred_bboxes'])[flag_instance==1]
        ret_dict['pred_scores'] = torch.tensor(pred_data['pred_scores'])[flag_instance==1]
        ret_dict['index'] = torch.IntTensor([index])
        return ret_dict

        