"""
#   Copyright (C) 2022 BAIDU CORPORATION. All rights reserved.
#   Author     :   tanglicheng@baidu.com
#   Date       :   2022-02-10
"""
from .base_trainer import BaseTrainer
import utils
import paddle
from itertools import chain


class CombinedTrainer(BaseTrainer):
    """
    CombinedTrainer
    """
    def __init__(self, gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                 logger, evaluator, cv_loaders, cfg): # cls_char
        super().__init__(gen, disc, g_optim, d_optim, g_scheduler, d_scheduler,
                         logger, evaluator, cv_loaders, cfg)

    def train(self, loader, st_step=1, max_step=100000):
        """
        train
        """
        self.gen.train()
        if self.disc is not None:
            self.disc.train()

        # loss stats
        losses = utils.AverageMeters("g_total", "pixel", "disc", "gen")
        # discriminator stats
        discs = utils.AverageMeters("real_font", "real_uni", "fake_font", "fake_uni")
        # etc stats
        stats = utils.AverageMeters("B_style", "B_target")
        self.step = st_step
        self.clear_losses()
        self.logger.info("Start training ...")
        
        while True:
            for (in_style_ids, in_imgs,
                 trg_style_ids, trg_uni_ids, trg_imgs, content_imgs, trg_unis, style_sample_index, trg_sample_index) in loader():

                epoch = self.step // len(loader)
                B = trg_imgs.shape[0]
                stats.updates({
                    "B_style": in_imgs.shape[0],
                    "B_target": B
                })

                # uni_ids are only used for decompose
                in_style_ids = in_style_ids.cuda()
                in_imgs = in_imgs.cuda()
                trg_uni_disc_ids = trg_uni_ids.cuda()
                trg_style_ids = trg_style_ids.cuda()
                trg_imgs = trg_imgs.cuda()
                content_imgs = content_imgs.cuda()

                if self.cfg.use_half:
                    in_imgs = in_imgs.half()
                    content_imgs = content_imgs.half()

                ##############################################################
                # infer
                ##############################################################
                sc_feats = self.gen.encode_write_comb(in_style_ids, style_sample_index, in_imgs)
                out, feat_main = self.gen.read_decode(trg_style_ids, trg_sample_index, content_imgs) #fake_img

                #reconstruction
                #self_infer_imgs = 0.0
                self_infer_imgs, feat_recons = self.gen.infer(trg_style_ids, trg_imgs, trg_style_ids, trg_sample_index, trg_sample_index, content_imgs)

                real_font, real_uni = self.disc(trg_imgs, trg_style_ids, trg_uni_disc_ids)
                fake_font, fake_uni = self.disc(out.detach(), trg_style_ids, trg_uni_disc_ids)

                #fake_font_recon, fake_uni_recon = 0, 0
                fake_font_recon, fake_uni_recon = self.disc(self_infer_imgs.detach(), trg_style_ids, trg_uni_disc_ids)
                self.add_gan_d_loss(real_font, real_uni, fake_font+fake_font_recon, fake_uni+fake_uni_recon)

                self.d_backward()
                self.d_optim.step()
                self.d_scheduler.step()
                self.d_optim.clear_grad()

                ################### generator ##################
                fake_font, fake_uni = self.disc(out, trg_style_ids, trg_uni_disc_ids)

                #reconstruction
                #fake_font_recon, fake_uni_recon = 0, 0
                fake_font_recon, fake_uni_recon = self.disc(self_infer_imgs, trg_style_ids, trg_uni_disc_ids)
                self.add_gan_g_loss(real_font, real_uni, fake_font+fake_font_recon, fake_uni+fake_uni_recon)
                self.add_pixel_loss(out, trg_imgs, self_infer_imgs)

                self.g_backward()
                self.g_optim.step()
                self.g_scheduler.step()
                self.g_optim.clear_grad()


                discs.updates({
                    "real_font": real_font.mean().item(),
                    "real_uni": real_uni.mean().item(),
                    "fake_font": fake_font.mean().item(),
                    "fake_uni": fake_uni.mean().item(),
                }, B)

                loss_dic = self.clear_losses()
                losses.updates(loss_dic, B)  # accum loss stats

                # EMA g
                self.accum_g()
                if self.step % self.cfg['tb_freq'] == 0:
                    self.baseplot(losses, discs, stats)

                if self.step % self.cfg['print_freq'] == 0:
                    self.log(losses, discs, stats)
                    self.logger.debug("GPU Memory usage: max mem_alloc = %.1fM / %.1fM",
                                      paddle.device.cuda.max_memory_allocated() / 1000 / 1000,
                                      paddle.device.cuda.max_memory_reserved() / 1000 / 1000)
                    losses.resets()
                    discs.resets()
                    stats.resets()

                if self.step % self.cfg['val_freq'] == 0:
                    epoch = self.step / len(loader)
                    self.logger.info("Validation at Epoch = {:.3f}".format(epoch))
                    self.evaluator.cp_validation(self.gen_ema, self.cv_loaders, self.step)
                    self.save(loss_dic['g_total'], self.cfg['save'], self.cfg.get('save_freq', self.cfg['val_freq']))

                if self.step >= max_step:
                    break

                self.step += 1
                
            if self.step >= max_step:
                break
            
        self.logger.info("Iteration finished.")

    def log(self, losses, discs, stats):
        self.logger.info(
            "  Step {step:7d}: L1 {L.pixel.avg:7.4f}  D {L.disc.avg:7.3f}  G {L.gen.avg:7.3f}"
            "  B_stl {S.B_style.avg:5.1f}  B_trg {S.B_target.avg:5.1f}"
            .format(step=self.step, L=losses, D=discs, S=stats))
