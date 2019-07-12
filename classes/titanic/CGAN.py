from classes.titanic.netG import netG
from classes.titanic.netD import netD


class CGAN:
    def __init__(self, device, nc, nz, sched_netG, path, x_dim, out_dim,
                 netG_H, netG_lr, netG_beta1, netG_beta2, netG_wd,
                 netD_H, netD_lr, netD_beta1, netD_beta2, netD_wd,
                 eval_param_grid,
                 cont_inputs, cat_inputs, int_inputs, preprocessed_cat_mask, le_dict):
        # Initialize properties
        self.device = device
        self.nc = nc
        self.nz = nz
        self.sched_netG = sched_netG,
        self.path = path
        self.x_dim = x_dim
        self.cont_inputs = cont_inputs
        self.cat_inputs = cat_inputs
        self.int_inputs = int_inputs
        self.preprocessed_cat_mask = preprocessed_cat_mask
        self.le_dict = le_dict
        self.eval_param_grid = eval_param_grid

        # Instantiate sub-nets
        self.netG = netG(nz=self.nz, H=netG_H, out_dim=out_dim, nc=self.nc, device=self.device,
                         wd=netG_wd, cat_mask=self.preprocessed_cat_mask, le_dict=self.le_dict,
                         lr=netG_lr, beta1=netG_beta1, beta2=netG_beta2).to(device)
        self.netD = netD(H=netD_H, out_dim=out_dim, nc=self.nc, device=self.device, wd=netD_wd,
                         lr=netD_lr, beta1=netD_beta1, beta2=netD_beta2).to(device)
