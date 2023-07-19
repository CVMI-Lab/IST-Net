import os
import sys
import argparse
import logging
import random

import torch
import gorilla

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'provider'))
sys.path.append(os.path.join(BASE_DIR, 'model'))
sys.path.append(os.path.join(BASE_DIR, 'model', 'pointnet2'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

# from DPDN import Net
from solver import test_func, get_logger
from dataset import TestDataset
from evaluation_utils import evaluate

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
    parser.add_argument("--test_epoch",
                        type=int,
                        default=30,
                        help="test epoch")
    parser.add_argument('--mask_label', action='store_true', default=False,
                        help='whether having mask labels of real data')
    parser.add_argument('--only_eval', action='store_true', default=False,
                        help='whether directly evaluating the results')
    args_cfg = parser.parse_args()
    return args_cfg

def init():
    args = get_parser()
    exp_name = args.config.split("/")[-1].split(".")[0]
    # if args.test_path is not None:
    #     log_dir = args.test_path
    # else:
    #     log_dir = os.path.join("log", exp_name)

    cfg = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    if cfg.test.test_path is not None:
        log_dir = cfg.test.test_path
    else:
        log_dir = os.path.join("log", exp_name)
    cfg.log_dir = log_dir
    cfg.gpus = args.gpus
    cfg.test_epoch = args.test_epoch
    cfg.mask_label = args.mask_label
    cfg.only_eval = args.only_eval

    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)
    logger = get_logger(level_print=logging.INFO, level_save=logging.WARNING, path_file=log_dir+"/test_epoch" + str(cfg.test_epoch)  + "_logger.log")

    return logger, cfg

if __name__ == "__main__":
    logger, cfg = init()

    logger.warning("************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    save_path = os.path.join(cfg.log_dir, 'eval_epoch' + str(cfg.test_epoch))
    if not cfg.only_eval:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        # model
        logger.info("=> creating model ...")
        if cfg.model_arch == "ist_net":
            from ist_net import IST_Net
            model = IST_Net(cfg.num_category, cfg.freeze_world_enhancer)

        if len(cfg.gpus)>1:
            model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
        model = model.cuda()

        checkpoint = os.path.join(cfg.log_dir, 'epoch_' + str(cfg.test_epoch) + '.pth')
        logger.info("=> loading checkpoint from path: {} ...".format(checkpoint))
        gorilla.solver.load_checkpoint(model=model, filename=checkpoint)

        # data loader

        TestingDataset = TestDataset
        dataset = TestingDataset(cfg.test, BASE_DIR)
        dataloder = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                num_workers=8,
                shuffle=False,
                drop_last=False
            )
        test_func(model, dataloder, save_path)

    evaluate(save_path, logger)

