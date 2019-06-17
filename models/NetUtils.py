import torch as nn


# Contains utils to be inherited by other nets in this project
class NetUtils:
    def init_hist(self):
        assert self.layer_list, "Make sure to initialize a self.layer_list in your constructor consisting of a list of layers to track to use this method."
        self.gnorm_hist = {}
        self.gnorm_total_hist = []
        self.wnorm_hist = {}
        self.wnorm_total_hist = []
        for layer in self.layer_list:
            self.gnorm_hist[layer] = {'weight': [], 'bias': []}
            self.wnorm_hist[layer] = {'weight': [], 'bias': []}

    def update_gnormz(self, norm_num):
        """
        Calculates gradient norms by layer as well as overall
        :param norm_num: 1 = l1 norm, 2 = l2 norm
        :return: list of gradient norms by layer, as well as overall gradient norm
        """
        total_norm = 0
        for layer in self.gnorm_hist:
            w_norm = layer.weight.grad.norm(norm_num).detach().cpu().numpy().take(0)
            b_norm = layer.bias.grad.norm(norm_num).detach().cpu().numpy().take(0)
            self.gnorm_hist[layer]['weight'].append(w_norm)
            self.gnorm_hist[layer]['bias'].append(b_norm)
            if norm_num == 1:
                total_norm += abs(w_norm) + abs(b_norm)
            else:
                total_norm += w_norm**norm_num + b_norm**norm_num
        total_norm = total_norm**(1./norm_num)
        self.gnorm_total_hist.append(total_norm)

    def update_wnormz(self, norm_num):
        """
        Tracks history of desired norm of weights
        :param norm_num: 1 = l1 norm, 2 = l2 norm
        :return: list of norms of weights by layer, as well as overall weight norm
        """
        total_norm = 0
        for layer in self.wnorm_hist:
            w_norm = layer.weight.norm(norm_num).detach().cpu().numpy().take(0)
            b_norm = layer.bias.norm(norm_num).detach().cpu().numpy().take(0)
            self.wnorm_hist[layer]['weight'].append(w_norm)
            self.wnorm_hist[layer]['bias'].append(b_norm)
            if norm_num == 1:
                total_norm += abs(w_norm) + abs(b_norm)
            else:
                total_norm += w_norm ** norm_num + b_norm ** norm_num
        total_norm = total_norm ** (1. / norm_num)
        self.wnorm_total_hist.append(total_norm)

    # Custom weights initialization called on Generator and Discriminator
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:  # TODO: May need to mess around with this later
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
