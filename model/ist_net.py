import torch
import torch.nn as nn
import torch.nn.functional as F

from rotation_utils import Ortho6d2Mat

from modules import ModifiedResnet, PointNet2MSG, PoseNet
from losses import SmoothL1Dis, ChamferDis, PoseDis

class IST_Net(nn.Module):
    def __init__(self, nclass=6):
        super(IST_Net, self).__init__()
        self.nclass = nclass
        self.rgb_cam_extractor = ModifiedResnet()
        self.pts_cam_extractor = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02,0.04], [0.04,0.08], [0.08,0.16]])
        self.pts_world_extractor = PointNet2MSG(radii_list=[[0.05,0.10], [0.10,0.20], [0.20,0.30], [0.30,0.40]])
        self.implicit_transform = ImplicitTransformation(nclass)
        self.main_estimator = HeavyEstimator()
        self.cam_enhancer = LightEstimator()
        self.world_enhancer = HeavyEstimator()

    def forward(self, inputs):
        end_points = {}

        # assert False
        rgb = inputs['rgb']
        pts = inputs['pts']
        choose = inputs['choose']
        # prior = inputs['prior']
        if self.training:
            pts_w_gt = inputs['qo']
        cls = inputs['category_label'].reshape(-1)

        c = torch.mean(pts, 1, keepdim=True)
        pts = pts - c

        b = pts.size(0)
        index = cls + torch.arange(b, dtype=torch.long).cuda() * self.nclass

        # rgb feat
        rgb_local = self.rgb_cam_extractor(rgb) # extract image level feature
        d = rgb_local.size(1)
        rgb_local = rgb_local.view(b, d, -1)
        choose = choose.unsqueeze(1).repeat(1, d, 1)
        rgb_local = torch.gather(rgb_local, 2, choose).contiguous()

        if self.training:
            pts_local = self.pts_cam_extractor(pts)
            pts_w_local_gt = self.pts_world_extractor(pts_w_gt)
            r_aux_cam, t_aux_cam, s_aux_cam = self.cam_enhancer(pts, rgb_local, pts_local)
            pts_w, pts_w_local = self.implicit_transform(rgb_local, pts_local, pts, c, index)
            r, t, s = self.main_estimator(pts, pts_w, rgb_local, pts_local, pts_w_local)
            r_aux_world, t_aux_world, s_aux_world = self.world_enhancer(pts, pts_w_gt, rgb_local.detach(), pts_local.detach(), pts_w_local_gt)
            end_points["pred_qo"] = pts_w
            end_points["pts_w_local"] = pts_w_local
            end_points["pts_w_local_gt"] = pts_w_local_gt
            end_points['pred_rotation'] = r
            end_points['pred_translation'] = t + c.squeeze(1)
            end_points['pred_size'] = s
            end_points['pred_rotation_aux_cam'] = r_aux_cam
            end_points['pred_translation_aux_cam'] = t_aux_cam + c.squeeze(1)
            end_points['pred_size_aux_cam'] = s_aux_cam
            end_points['pred_rotation_aux_world'] = r_aux_world
            end_points['pred_translation_aux_world'] = t_aux_world + c.squeeze(1)
            end_points['pred_size_aux_world'] = s_aux_world
        else:
            pts_local = self.pts_cam_extractor(pts)
            pts_w, pts_w_local = self.implicit_transform(rgb_local, pts_local, pts, c, index)
            r, t, s = self.main_estimator(pts, pts_w, rgb_local, pts_local, pts_w_local)
            end_points["pred_qo"] = pts_w
            end_points['pred_rotation'] = r
            end_points['pred_translation'] = t + c.squeeze(1)
            end_points['pred_size'] = s

        return end_points

class SupervisedLoss(nn.Module):
    def __init__(self, cfg):
        super(SupervisedLoss, self).__init__()
        self.cfg = cfg

    def forward(self, end_points):
        qo = end_points['pred_qo']
        t = end_points['pred_translation']
        r = end_points['pred_rotation']
        s = end_points['pred_size']
        loss = self._get_loss(r, t, s, qo, end_points)

        return loss
  
    def _get_loss(self, r, t, s, qo, end_points):
        pts_w_local = end_points["pts_w_local"]
        pts_w_local_gt = end_points["pts_w_local_gt"]
        t_aux_cam = end_points['pred_translation_aux_cam']
        r_aux_cam = end_points['pred_rotation_aux_cam']
        s_aux_cam = end_points['pred_size_aux_cam']
        r_aux_world = end_points['pred_rotation_aux_world']
        t_aux_world = end_points['pred_translation_aux_world']
        s_aux_world = end_points['pred_size_aux_world']
        loss_feat = nn.functional.mse_loss(pts_w_local, pts_w_local_gt)
        loss_qo = SmoothL1Dis(qo, end_points['qo'])
        loss_pose = PoseDis(r, t, s, end_points['rotation_label'],end_points['translation_label'],end_points['size_label'])
        loss_pose_aux_cam = PoseDis(r_aux_cam, t_aux_cam, s_aux_cam, end_points['rotation_label'],end_points['translation_label'], end_points['size_label'])
        loss_pose_aux_world = PoseDis(r_aux_world, t_aux_world, s_aux_world, end_points['rotation_label'],end_points['translation_label'], end_points['size_label'])
        cfg = self.cfg
        loss = loss_pose + loss_pose_aux_cam + loss_pose_aux_world + cfg.gamma2 * loss_qo + cfg.gamma3*loss_feat
        return loss


class ImplicitTransformation(nn.Module):
    def __init__(self, nclass=6):
        super(ImplicitTransformation, self).__init__()
        self.nclass = nclass
        self.feature_refine = FeatureDeformer(nclass)

    def forward(self, rgb_local, pts_local, pts, center, index):
        pts_local_w, pts_w = self.feature_refine(pts, rgb_local, pts_local, index)
        return pts_w, pts_local_w


class FeatureDeformer(nn.Module):
    def __init__(self, nclass=6):
        super(FeatureDeformer, self).__init__()
        self.nclass = nclass

        self.pts_mlp1 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )

        self.deform_mlp1 = nn.Sequential(
            nn.Conv1d(64+256, 384, 1),
            nn.ReLU(),
            nn.Conv1d(384, 256, 1),
            nn.ReLU(),
        )

        self.deform_mlp2 = nn.Sequential(
            nn.Conv1d(512, 384, 1),
            nn.ReLU(),
            nn.Conv1d(384, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
        )


        self.pred_nocs = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, nclass*3, 1),
        )

    def forward(self, pts, rgb_local, pts_local, index):

        npoint = pts_local.size(2)
        pts_pose_feat = self.pts_mlp1(pts.transpose(1,2))

        deform_feat = torch.cat([
            pts_pose_feat,
            pts_local,
            rgb_local,
        ], dim=1)

        pts_local_w = self.deform_mlp1(deform_feat)
        pts_global_w = torch.mean(pts_local_w, 2, keepdim=True)
        pts_local_w = torch.cat([pts_local_w, pts_global_w.expand_as(pts_local_w)], 1)
        pts_local_w = self.deform_mlp2(pts_local_w)

        pts_w = self.pred_nocs(pts_local_w)
        pts_w = pts_w.view(-1, 3, npoint).contiguous()
        pts_w = torch.index_select(pts_w, 0, index)   # bs x 3 x nv
        pts_w = pts_w.permute(0, 2, 1).contiguous()   # bs x nv x 3

        return pts_local_w, pts_w


class LightEstimator(nn.Module):
    def __init__(self):
        super(LightEstimator, self).__init__()

        self.pts_mlp = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )

        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(128+64+128, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
        )

        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.rotation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )
        self.translation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.size_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, pts, rgb_local, pts_local):

        pts = self.pts_mlp(pts.transpose(1,2))
        pose_feat = torch.cat([rgb_local, pts, pts_local], dim=1) # 

        pose_feat = self.pose_mlp1(pose_feat)
        pose_global = torch.mean(pose_feat, 2, keepdim=True)
        pose_feat = torch.cat([pose_feat, pose_global.expand_as(pose_feat)], 1)
        pose_feat = self.pose_mlp2(pose_feat).squeeze(2)

        r = self.rotation_estimator(pose_feat)
        r = Ortho6d2Mat(r[:, :3].contiguous(), r[:, 3:].contiguous()).view(-1,3,3)
        t = self.translation_estimator(pose_feat)
        s = self.size_estimator(pose_feat)
        return r,t,s

    
class HeavyEstimator(nn.Module):
    def __init__(self):
        super(HeavyEstimator, self).__init__()

        self.pts_mlp1 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pts_mlp2 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(64+64+384, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
        )
        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.rotation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )
        self.translation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.size_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, pts, pts_w, rgb_local, pts_local, pts_w_local):
        pts = self.pts_mlp1(pts.transpose(1,2))
        pts_w = self.pts_mlp2(pts_w.transpose(1,2))

        pose_feat = torch.cat([rgb_local, pts, pts_local, pts_w, pts_w_local], dim=1)
        pose_feat = self.pose_mlp1(pose_feat)
        pose_global = torch.mean(pose_feat, 2, keepdim=True)
        pose_feat = torch.cat([pose_feat, pose_global.expand_as(pose_feat)], 1)
        pose_feat = self.pose_mlp2(pose_feat).squeeze(2)

        r = self.rotation_estimator(pose_feat)
        r = Ortho6d2Mat(r[:, :3].contiguous(), r[:, 3:].contiguous()).view(-1,3,3)
        t = self.translation_estimator(pose_feat)
        s = self.size_estimator(pose_feat)
        return r,t,s