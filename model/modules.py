import torch
import torch.nn as nn
import torch.nn.functional as F

import resnet
from pointnet2_modules import PointnetSAModuleMSG, PointnetFPModule
from rotation_utils import Ortho6d2Mat


class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(features, size) for size in sizes]
        )
        self.bottleneck = nn.Conv2d(
            features * (len(sizes) + 1), out_features, kernel_size=1
        )
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.upsample(input=stage(feats), size=(h, w), mode='bilinear')
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


class Modified_PSPNet(nn.Module):
    def __init__(self, sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet18', pretrained=True):
        super(Modified_PSPNet, self).__init__()
        self.feats = getattr(resnet, backend)(pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        return self.final(p)


class DeepPriorDeformer(nn.Module):
    def __init__(self, nclass=6, nprior=1024):
        super(DeepPriorDeformer, self).__init__()
        self.nclass = nclass
        self.nprior = nprior

        self.atten_mlp = nn.Sequential(
            nn.Conv1d(384, 384, 1),
            nn.ReLU(),
            nn.Conv1d(384, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, nclass*nprior, 1),
        )
        self.deform_mlp1 = nn.Sequential(
            nn.Conv1d(384, 384, 1),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Conv1d(384, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
        )
        self.deform_mlp2 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, nclass*3, 1),
        )

    def forward(self, rgb_local, pts_local, prior_local, index):
        """
        rgb_local 提取过特征的 rgb feature
        pts_local 提取过特征的 depth feature
        prior_local 提取过特征的 prior feature
        """
        nprior = self.nprior # prior point 的个数
        npoint = pts_local.size(2)

        # deform in feature space
        rgb_global = torch.mean(rgb_local, 2, keepdim=True)
        pts_global = torch.mean(pts_local, 2, keepdim=True)
        deform_feat = torch.cat([
            prior_local,
            pts_global.repeat(1, 1, nprior),
            rgb_global.repeat(1, 1, nprior)
        ], dim=1)
        deform_feat = self.deform_mlp1(deform_feat)

        prior_local = F.relu(prior_local + deform_feat)
        prior_global = torch.mean(prior_local, 2, keepdim=True)

        ## attention
        atten_feat = torch.cat((pts_local, rgb_local, prior_global.repeat(1, 1, npoint)), dim=1)
        atten_feat = self.atten_mlp(atten_feat)
        atten_feat = atten_feat.view(-1, nprior, npoint).contiguous()
        atten_feat = torch.index_select(atten_feat, 0, index)
        attention = atten_feat.permute(0, 2, 1).contiguous()    # b x npoint x nprior

        ## Qv, Qo
        Qv = self.deform_mlp2(prior_local)
        Qv = Qv.view(-1, 3, nprior).contiguous() 
        Qv = torch.index_select(Qv, 0, index)
        Qv = Qv.permute(0, 2, 1).contiguous()   # b x nprior x 3

        Qo = torch.bmm(Qv.transpose(1,2).detach(), F.softmax(attention.detach(), dim=2).permute(0, 2, 1).contiguous())
        new_prior_local = torch.bmm(prior_local, F.softmax(attention, dim=2).permute(0, 2, 1).contiguous())
        return attention, new_prior_local, Qv, Qo.transpose(1,2)


class PoseSizeEstimator(nn.Module):
    def __init__(self):
        super(PoseSizeEstimator, self).__init__()
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

    def forward(self, pts1, pts2, rgb_local, pts1_local, pts2_local):
        pts1 = self.pts_mlp1(pts1.transpose(1,2))
        pts2 = self.pts_mlp2(pts2.transpose(1,2))
        pose_feat = torch.cat([rgb_local, pts1, pts1_local, pts2, pts2_local], dim=1)

        pose_feat = self.pose_mlp1(pose_feat) # [5, 256, 1024]
        pose_global = torch.mean(pose_feat, 2, keepdim=True)
        pose_feat = torch.cat([pose_feat, pose_global.expand_as(pose_feat)], 1)
        pose_feat = self.pose_mlp2(pose_feat).squeeze(2)
        # print("pose_feat after:", pose_feat.shape)
        r = self.rotation_estimator(pose_feat)
        # print("pose_feat shape:", pose_feat.shape, "r shape:", r.shape)
        r = Ortho6d2Mat(r[:, :3].contiguous(), r[:, 3:].contiguous()).view(-1,3,3)
        t = self.translation_estimator(pose_feat)
        s = self.size_estimator(pose_feat)
        return r,t,s


# main modules

modified_psp_models = {
    'resnet18': lambda: Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=512, backend='resnet18'),
    'resnet34': lambda: Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=512, backend='resnet34'),
    'resnet50': lambda: Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet50'),
    'resnet101': lambda:Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet101'),
    'resnet152': lambda:Modified_PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, backend='resnet152')
}


class ModifiedResnet(nn.Module):
    def __init__(self):
        super(ModifiedResnet, self).__init__()
        self.model = modified_psp_models['resnet18'.lower()]()

    def forward(self, x):
        x = self.model(x)
        return x


class PointNet2MSG(nn.Module):
    def __init__(self, radii_list, use_xyz=True):
        super(PointNet2MSG, self).__init__()
        self.SA_modules = nn.ModuleList()
        c_in = 0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=radii_list[0],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 16, 16, 32]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_0 = 32 + 32

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=256,
                radii=radii_list[1],
                nsamples=[16, 32],
                mlps=[[c_in, 32, 32, 64], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_1 = 64 + 64

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=radii_list[2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 64, 128]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_2 = 128 + 128

        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=64,
                radii=radii_list[3],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 128, 256], [c_in, 128, 128, 256]],
                use_xyz=use_xyz,
                bn=True,
            )
        )
        c_out_3 = 256 + 256

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256, 128, 128], bn=True))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_out_0, 256, 256], bn=True))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 256, 256], bn=True))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512], bn=True))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud):
        _, N, _ = pointcloud.size()

        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return l_features[0]


class PoseNet(nn.Module):
    def __init__(self, nclass=6, nprior=1024):
        super(PoseNet, self).__init__()
        self.deformer = DeepPriorDeformer(nclass, nprior)
        self.estimator = PoseSizeEstimator()

    def forward(self, rgb_local, pts_local, prior_local, pts, index):
        A, new_prior_local, Qv, Qo = self.deformer(rgb_local, pts_local, prior_local, index)
        r,t,s = self.estimator(pts, Qo, rgb_local, pts_local, new_prior_local)
        return A, Qv, r, t, s
