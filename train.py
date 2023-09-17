import os
import sys
import argparse
import logging
import random
import collections
import torch
import gorilla

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

from solver import Solver, get_logger
from dataset import TrainingDataset
# from DPDN import Net, SupervisedLoss, UnSupervisedLoss

def get_parser():
    parser = argparse.ArgumentParser(
        description="Pose Estimation")

    # pretrain
    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="gpu num")
    parser.add_argument("--config",
                        type=str,
                        default="config/supervised.yaml",
                        help="path to config file")
    parser.add_argument("--checkpoint_epoch",
                        type=int,
                        default=-1,
                        help="checkpoint epoch")
    args_cfg = parser.parse_args()

    return args_cfg

def init():
    args = get_parser()
    exp_name = args.config.split("/")[-1].split(".")[0]
    log_dir = os.path.join("log", exp_name)
    if not os.path.isdir("log"):
        os.makedirs("log")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.log_dir = log_dir
    cfg.gpus = args.gpus
    cfg.checkpoint_epoch = args.checkpoint_epoch

    logger = get_logger(
        level_print=logging.INFO, level_save=logging.WARNING, path_file=log_dir+"/training_logger.log")
    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)

    return logger, cfg


if __name__ == "__main__":
    logger, cfg = init()
    

    logger.warning(
        "************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    # Get model
    logger.info("=> creating model ...")
    if cfg.model_arch == "ist_net":
        from ist_net import IST_Net, SupervisedLoss
        model = IST_Net(cfg.num_category, cfg.freeze_world_enhancer)
    elif cfg.model_arch == "posenet_gt":
        from posenet_gt import PoseNetGT, SupervisedLoss
        model = PoseNetGT(cfg.num_category)
    else:
        raise Exception('architecture {} not supported yet'.format(cfg.model_arch))


    if cfg.checkpoint_epoch != -1:
        logger.info("=> loading checkpoint from epoch {} ...".format(cfg.checkpoint_epoch))
        checkpoint = os.path.join(cfg.log_dir, 'epoch_' + str(cfg.checkpoint_epoch) + '.pth')
        checkpoint_file = gorilla.solver.load_checkpoint(model=model, filename=checkpoint)
        start_epoch = checkpoint_file['meta']['epoch']+1
        start_iter = checkpoint_file['meta']['iter']
        del checkpoint_file
    else:
        start_epoch = 1
        start_iter = 0

    if len(cfg.gpus) > 1:
        model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
    model = model.cuda()

    # load freezed world enhancer if needed
    if cfg.checkpoint_epoch == -1 and cfg.get("freeze_world_enhancer", False):
        assert cfg.world_enhancer_weights is not None
        checkpoint = torch.load(cfg.world_enhancer_weights, map_location=lambda storage, loc: storage.cuda())
        world_enhancer_dict = collections.OrderedDict()
        for k, v in checkpoint["model"].items():
            if "pts_gt_extractor" in k:
                if len(cfg.gpus) > 1:
                    new_k = k.replace("pts_gt_extractor.", "module.world_enhancer.extractor.")
                else:
                    new_k = k.replace("pts_gt_extractor.", "world_enhancer.extractor.")
                world_enhancer_dict[new_k] = v
        model.load_state_dict(world_enhancer_dict, strict=False)
        # set grads to False
        for name, param in model.named_parameters():
            if "world_enhancer" in name:
                param.requires_grad = False


    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.warning("#Total parameters : {}".format(count_parameters))

    # Get Loss
    loss_syn = SupervisedLoss(cfg).cuda()
    loss_real = SupervisedLoss(cfg).cuda()
    loss  = {
        "syn": loss_syn,
        "real": loss_real,
    }


    # dataloader
    data_mode = cfg.get("data_mode", "Camera+Real")
    data_dir = os.path.join(BASE_DIR, 'data')
    RealDataset = TrainingDataset
    SynDataset = TrainingDataset

    if "Camera" in data_mode:
        syn_dataset = SynDataset(
            cfg.train_dataset, data_dir, 'syn',
            num_img_per_epoch=cfg.num_mini_batch_per_epoch*cfg.train_dataloader.syn_bs,
            use_fill_miss = cfg.train_dataloader.use_fill_miss,
            use_composed_img = cfg.train_dataloader.use_composed_img,
            per_obj=cfg.train_dataloader.per_obj)
        syn_dataloader = torch.utils.data.DataLoader(
            syn_dataset,
            batch_size=cfg.train_dataloader.syn_bs,
            num_workers=cfg.train_dataloader.num_workers,
            shuffle=cfg.train_dataloader.shuffle,
            sampler=None,
            drop_last=cfg.train_dataloader.drop_last,
            pin_memory=cfg.train_dataloader.pin_memory
        )
    else:
        syn_dataloader = None

    if "Real" in data_mode:
        data_type = 'real_withLabel'

        real_dataset = RealDataset(
            cfg.train_dataset, data_dir, data_type,
            num_img_per_epoch=cfg.num_mini_batch_per_epoch*cfg.train_dataloader.real_bs,
            use_fill_miss = cfg.train_dataloader.use_fill_miss,
            use_composed_img = cfg.train_dataloader.use_composed_img,
            per_obj=cfg.train_dataloader.per_obj)
        real_dataloader = torch.utils.data.DataLoader(
            real_dataset,
            batch_size=cfg.train_dataloader.real_bs,
            num_workers=cfg.train_dataloader.num_workers,
            shuffle=cfg.train_dataloader.shuffle,
            sampler=None,
            drop_last=cfg.train_dataloader.drop_last,
            pin_memory=cfg.train_dataloader.pin_memory
        )
    else:
        real_dataloader = None

    dataloaders = {
        "syn": syn_dataloader,
        "real": real_dataloader,
    }

    # solver
    Trainer = Solver(model=model, data_mode=data_mode, loss=loss,
                     dataloaders=dataloaders,
                     logger=logger,
                     cfg=cfg,
                     start_epoch=start_epoch,
                     start_iter=start_iter)
    Trainer.solve()

    logger.info('\nFinish!\n')
