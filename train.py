"""
#   Copyright (C) 2022 BAIDU CORPORATION. All rights reserved.
#   Author     :   tanglicheng@baidu.com
#   Date       :   2022-02-10
"""
import sys
from pathlib import Path
import argparse
import paddle
import paddle.optimizer as optim
from paddle.vision import transforms
import numpy as np
from sconf import Config, dump_args
import utils
from utils import Logger
from transform import setup_transforms
from models import generator_dispatch, disc_builder
from datasets import (load_lmdb, load_json, read_data_from_lmdb,
                      get_comb_trn_loader, get_cv_comb_loaders)
from trainer import load_checkpoint, CombinedTrainer
from evaluator import Evaluator
import copy


def setup_args_and_config():
    """
    setup_args_and_configs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("config_paths", nargs="+", help="path/to/config.yaml")
    parser.add_argument("--resume", default=None, help="path/to/saved/.pth")
    parser.add_argument("--use_unique_name", default=False, action="store_true", help="whether to use name with timestamp")

    args, left_argv = parser.parse_known_args()
    assert not args.name.endswith(".yaml")

    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml",
                 colorize_modified_item=True)
    cfg.argv_update(left_argv)

    cfg.work_dir = Path(cfg.work_dir)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    if args.use_unique_name:
        timestamp = utils.timestamp()
        unique_name = "{}_{}".format(timestamp, args.name)
    else:
        unique_name = args.name

    cfg.unique_name = unique_name
    cfg.name = args.name

    (cfg.work_dir / "logs").mkdir(parents=True, exist_ok=True)
    (cfg.work_dir / "checkpoints" / unique_name).mkdir(parents=True, exist_ok=True)

    if cfg.save_freq % cfg.val_freq:
        raise ValueError("save_freq has to be multiple of val_freq.")

    return args, cfg


def train(args, cfg, ddp_gpu=-1):
    """
    train
    """
    paddle.device.set_device('gpu')
    logger_path = cfg.work_dir / "logs" / "{}.log".format(cfg.unique_name)
    logger = Logger.get(file_path=logger_path, level="info", colorize=True)

    image_scale = 0.6
    writer_path = cfg.work_dir / "runs" / cfg.unique_name
    eval_image_path = cfg.work_dir / "images" / cfg.unique_name
    writer = utils.TBDiskWriter(writer_path, eval_image_path, scale=image_scale)

    args_str = dump_args(args)
    #if is_main_worker(ddp_gpu):
    logger.info("Run Argv:\n> {}".format(" ".join(sys.argv)))
    logger.info("Args:\n{}".format(args_str))
    logger.info("Configs:\n{}".format(cfg.dumps()))
    logger.info("Unique name: {}".format(cfg.unique_name))
    logger.info("Get dataset ...")

    content_font = cfg.content_font

    trn_transform, val_transform = setup_transforms(cfg)

    env = load_lmdb(cfg.data_path)
    env_get = lambda env, x, y, transform: transform(read_data_from_lmdb(env, f'{x}_{y}')['img'])
    data_meta = load_json(cfg.data_meta)

    get_trn_loader = get_comb_trn_loader
    get_cv_loaders = get_cv_comb_loaders
    Trainer = CombinedTrainer

    trn_dset, trn_loader = get_trn_loader(env,
                              env_get,
                              cfg,
                              data_meta["train"],
                              trn_transform,
                              num_workers=cfg.n_workers,
                              shuffle=False)

    cv_loaders = get_cv_loaders(env,
                    env_get,
                    cfg,
                    data_meta,
                    val_transform,
                    num_workers=cfg.n_workers,
                    shuffle=False)

    logger.info("Build model ...")
    # generator
    g_kwargs = cfg.get("g_args", {})
    g_cls = generator_dispatch()
    gen = g_cls(1, cfg.C, 1, cfg, **g_kwargs)
    gen.to('gpu')

    if cfg.gan_w > 0.:
        d_kwargs = cfg.get("d_args", {})
        disc = disc_builder(cfg.C, trn_dset.n_fonts, trn_dset.n_unis, **d_kwargs)
        disc.to('gpu')
    else:
        disc = None
    
    gen_scheduler = paddle.optimizer.lr.StepDecay(learning_rate=cfg.g_lr, step_size=cfg['step_size'], gamma=cfg['gamma'], verbose=False)
    dis_scheduler = paddle.optimizer.lr.StepDecay(learning_rate=cfg.d_lr, step_size=cfg['step_size'], gamma=cfg['gamma'], verbose=False)
    g_optim = optim.Adam(parameters=gen.parameters(),  learning_rate=gen_scheduler, weight_decay=0.0)
    d_optim = optim.Adam(parameters=disc.parameters(), learning_rate=dis_scheduler, weight_decay=0.0) \
        if disc is not None else None

    st_step = 1
    if args.resume:
        st_step, loss = load_checkpoint(args.resume, gen, disc, g_optim, d_optim, gen_scheduler, dis_scheduler)
        logger.info("Resumed checkpoint from {} (Step {}, Loss {:7.3f})".format(
            args.resume, st_step - 1, loss))
        if cfg.overwrite:
            st_step = 1
        else:
            pass

    evaluator = Evaluator(env,
                          env_get,
                          cfg,
                          logger,
                          writer,
                          cfg.batch_size,
                          val_transform,
                          content_font,
                          use_half=cfg.use_half
                          )

    trainer = Trainer(gen, disc, g_optim, d_optim, gen_scheduler, dis_scheduler,
                      logger, evaluator, cv_loaders, cfg)
    trainer.train(trn_loader, st_step, cfg["iter"])


def main():
    args, cfg = setup_args_and_config()
    np.random.seed(cfg["seed"])
    paddle.seed(cfg["seed"])
    train(args, cfg)

if __name__ == "__main__":
    main()
