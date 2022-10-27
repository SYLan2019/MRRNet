import torch
import torch.nn as nn
import torch.optim as optim

from models import loss
from models import networks
from .base_model import BaseModel
from utils import utils
from models.MRRNet import MRRNet

from models.loss import RaHingeGANLoss
from models.blocks import Discriminator


class MRRNetModel(BaseModel):

    def modify_commandline_options(parser, is_train):
        parser.add_argument('--scale_factor', type=int, default=8, help='upscale factor for mrrnet')
        parser.add_argument('--lambda_pix', type=float, default=1.0, help='weight for pixel loss')
        # parser.add_argument('--lambda_pix', type=float, default=0.01, help='weight for pixel loss')
        parser.add_argument('--lambda_G', type=float, default=0.01, help='weight for GAN-G')
        parser.add_argument('--lambda_pcp', type=float, default=0.01, help='weight for vgg perceptual loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.netG = MRRNet(res_depth=opt.res_depth, norm_type=opt.Gnorm, att_name=opt.att_name,
                            bottleneck_size=opt.bottleneck_size)
        self.netG = networks.define_network(opt, self.netG)

        self.model_names = ['G', 'D']  # <--
        self.load_model_names = ['G', 'D']
        self.loss_names = ['Pix', 'G', 'D', 'PCP']  # <--
        self.visual_names = ['img_LR', 'img_SR', 'img_HR']

        if self.isTrain:
            self.criterionL1 = nn.L1Loss()

            # 新增 RaHingeGan Loss perceptual loss
            self.criterionRaHingeGan = RaHingeGANLoss(gan_mode="rahinge")
            self.criterionPCP = loss.PCPLoss(opt)

            # 新增 discriminator
            self.netD = Discriminator(conv_dim=32, norm_fun="none", act_fun="LeakyReLU", use_sn=True,
                                      adv_loss_type="rahinge").to(opt.device)

            # 新增 vgg for perceptual loss
            self.vgg19 = loss.PCPFeat('./pretrain_models/vgg19-dcbb9e9d.pth', 'vgg')
            self.vgg19 = networks.define_network(opt, self.vgg19, isTrain=False, init_network=False)

            self.optimizer_G = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
            # 新增 optimizer_D
            self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.d_lr, betas=(opt.beta1, 0.99))
            self.optimizers = [self.optimizer_G, self.optimizer_D]
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_G, step_size=4, gamma=0.7)

    def load_pretrain_model(self, ):
        print('Loading pretrained model', self.opt.pretrain_model_path)
        weight = torch.load(self.opt.pretrain_model_path)
        self.netG.module.load_state_dict(weight)

    def set_input(self, input, cur_iters=None):
        self.cur_iters = cur_iters
        self.img_LR = input['LR'].to(self.opt.data_device)
        self.img_HR = input['HR'].to(self.opt.data_device)

    def forward(self):
        self.img_SR = self.netG(self.img_LR)

        # 新增
        self.real_pred = self.netD(self.img_HR)
        self.fake_pred = self.netD(self.img_SR)

        # for perceptual loss
        self.fake_vgg_feat = self.vgg19(self.img_SR)
        self.real_vgg_feat = self.vgg19(self.img_HR)

    def backward_G(self):
        # Pix loss
        self.loss_Pix = self.criterionL1(self.img_SR, self.img_HR) * self.opt.lambda_pix

        # 新增 RaHingeGan Loss
        self.loss_G = self.criterionRaHingeGan.loss(self.real_pred, self.fake_pred, for_discriminator=False,
                                                    target_is_real=True)

        # new perceptual loss
        self.loss_PCP = self.criterionPCP(self.fake_vgg_feat, self.real_vgg_feat)

        total_loss = self.opt.lambda_pix * self.loss_Pix + self.opt.lambda_G * self.loss_G + self.opt.lambda_pcp * self.loss_PCP

        # 改变
        # self.loss_Pix.backward()
        total_loss.backward()

    # 新增
    def backward_D(self):
        # 新增 RaHingeGan Loss for D
        # 新增
        self.img_SR = self.netG(self.img_LR).detach()
        self.real_pred = self.netD(self.img_HR)
        self.fake_pred = self.netD(self.img_SR)
        self.loss_D = self.criterionRaHingeGan.loss(self.real_pred, self.fake_pred, for_discriminator=True,
                                                    target_is_real=True)
        self.loss_D.backward()

    def optimize_parameters(self, ):
        # ---- Update G ------------
        self.network_grad_set(is_train_G=True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # ---- Update D-------------
        self.network_grad_set(is_train_G=False)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def update_lr(self):
        self.scheduler.step()

    # new!
    def network_grad_set(self, is_train_G):
        if is_train_G:
            for para in self.netD.parameters():
                para.requires_grad = False
        else:
            for para in self.netD.parameters():
                para.requires_grad = True


    def get_current_visuals(self, size=128):
        out = []
        out.append(utils.tensor_to_numpy(self.img_LR))
        out.append(utils.tensor_to_numpy(self.img_SR))
        out.append(utils.tensor_to_numpy(self.img_HR))
        visual_imgs = [utils.batch_numpy_to_image(x, size) for x in out]

        return visual_imgs
