import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import init
from torch.autograd import Function
#torch.manual_seed(42)


class L1(torch.nn.Module):
    def __init__(self, module, weight_decay):
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay

        # Backward hook is registered on the specified module
        self.hook = self.module.register_backward_hook(self._weight_decay_hook)

    # Not dependent on backprop incoming values, placeholder
    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            # If there is no gradient or it was zeroed out
            # Zeroed out using optimizer.zero_grad() usually
            # Turn on if needed with grad accumulation/more safer way
            # if param.grad is None or torch.all(param.grad == 0.0):

            # Apply regularization on it
            param.grad = self.regularize(param)

    def regularize(self, parameter):
        # L1 regularization formula
        return self.weight_decay * torch.sign(parameter.data)

    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)


class dummyhgru(Function):
    @staticmethod
    def forward(ctx, state_2nd_last, last_state, *args):
        ctx.save_for_backward(state_2nd_last, last_state)
        ctx.args = args
        return last_state

    @staticmethod
    def backward(ctx, grad):
        neumann_g = neumann_v = None
        neumann_g_prev = grad.clone()
        neumann_v_prev = grad.clone()

        state_2nd_last, last_state = ctx.saved_tensors
        
        args = ctx.args
        truncate_iter = args[-1]
        # exp_name = args[-2]
        # i = args[-3]
        # epoch = args[-4]

        normsv = []
        normsg = []
        normg = torch.norm(neumann_g_prev)
        # normsg.append(normg.data.item())
        # normsv.append(normg.data.item())
        prev_normv = 1e8
        for ii in range(truncate_iter):
            neumann_v = torch.autograd.grad(last_state, state_2nd_last, grad_outputs=neumann_v_prev,
                                            retain_graph=True, allow_unused=True)
            normv = torch.norm(neumann_v[0])
            neumann_g = neumann_g_prev + neumann_v[0]
            normg = torch.norm(neumann_g)
            
            if normg > 1 or normv > prev_normv or normv < 1e-9:
                # normsg.append(normg.data.item())
                # normsv.append(normv.data.item())
                neumann_g = neumann_g_prev
                break

            prev_normv = normv
            neumann_v_prev = neumann_v
            neumann_g_prev = neumann_g
            
            # normsv.append(normv.data.item())
            # normsg.append(normg.data.item())
        return (None, neumann_g, None, None, None, None)


class hConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, timesteps, batchnorm=True, grad_method='bptt', use_attention=False, no_inh=False, lesion_alpha=False, lesion_gamma=False, lesion_mu=False, lesion_kappa=False, att_nl=torch.sigmoid, l1=0.):
        super(hConvGRUCell, self).__init__()
        self.padding = kernel_size // 2
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batchnorm = batchnorm
        self.timesteps = timesteps
        self.use_attention = use_attention
        self.no_inh = no_inh
        self.att_nl = att_nl
        self.grad_method = grad_method
        
        if self.use_attention:
            self.a_w_gate = nn.Conv2d(hidden_size, hidden_size, 1, padding=1 // 2)
            self.a_u_gate = nn.Conv2d(hidden_size, hidden_size, 1, padding=1 // 2)
            init.orthogonal_(self.a_w_gate.weight)
            init.orthogonal_(self.a_u_gate.weight)
            init.constant_(self.a_w_gate.bias, 1.)  # In future try setting to -1 -- originally set to 1
            init.constant_(self.a_u_gate.bias, 1.)

        self.i_w_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.i_u_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        
        self.e_w_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.e_u_gate = nn.Conv2d(hidden_size, hidden_size, 1)

        self.exc_init = nn.Conv2d(hidden_size, hidden_size, 1)
        self.inh_init = nn.Conv2d(hidden_size, hidden_size, 1)

        spatial_h_size = kernel_size
        self.h_padding = spatial_h_size // 2
        # self.w_exc = nn.Parameter(torch.empty(hidden_size, hidden_size, spatial_h_size, spatial_h_size))
        # init.orthogonal_(self.w_exc)
        # if not no_inh:
        #     self.w_inh = nn.Parameter(torch.empty(hidden_size, hidden_size, spatial_h_size, spatial_h_size))
        #     init.orthogonal_(self.w_inh)
        self.w_exc = nn.Conv2d(hidden_size, hidden_size, spatial_h_size, padding=spatial_h_size//2)
        init.orthogonal_(self.w_exc.weight)
        if l1:
            self.w_exc = L1(self.w_exc, weight_decay=l1)
        if not no_inh:
            self.w_inh = nn.Conv2d(hidden_size, hidden_size, spatial_h_size, padding=spatial_h_size//2)
            init.orthogonal_(self.w_inh.weight) 
            if l1:
                self.w_inh = L1(self.w_inh, weight_decay=l1)

        self.alpha = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.mu = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        self.gamma = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size, 1, 1)))
        self.w = nn.Parameter(torch.empty((hidden_size, 1, 1)))

        self.bn = nn.ModuleList([nn.BatchNorm2d(hidden_size, eps=1e-03, affine=True, track_running_stats=False) for i in range(2)])
        # self.bn = nn.ModuleList([nn.GroupNorm(5, hidden_size, affine=True) for i in range(2)])
        self.gbn = nn.ModuleList([nn.GroupNorm(1, hidden_size, affine=True) for i in range(3)]) #  nn.BatchNorm2d(hidden_size, eps=1e-03, affine=True, track_running_stats=False) for i in range(2)])

        init.orthogonal_(self.i_w_gate.weight)
        init.orthogonal_(self.i_u_gate.weight)
        init.orthogonal_(self.e_w_gate.weight)
        init.orthogonal_(self.e_u_gate.weight)
        
        for bn, gbn in zip(self.bn, self.gbn):
            init.constant_(bn.weight, 0.1)
            init.constant_(gbn.weight, 0.1)

        if not no_inh:
            init.constant_(self.alpha, 1.)
            init.constant_(self.mu, 0.)
        # init.constant_(self.alpha, 0.1)
        # init.constant_(self.mu, 1)
        init.constant_(self.gamma, 0.)
        # init.constant_(self.w, 1.)
        init.constant_(self.kappa, 1.)

        if self.use_attention:
            self.i_w_gate.bias.data = -self.a_w_gate.bias.data
            self.e_w_gate.bias.data = -self.a_w_gate.bias.data
            self.i_u_gate.bias.data = -self.a_u_gate.bias.data
            self.e_u_gate.bias.data = -self.a_u_gate.bias.data
        else:
            init.uniform_(self.i_w_gate.bias.data, 1, self.timesteps - 1)
            self.i_w_gate.bias.data.log()
            self.i_u_gate.bias.data.log()
            self.e_w_gate.bias.data = -self.i_w_gate.bias.data
            self.e_u_gate.bias.data = -self.i_u_gate.bias.data
        if lesion_alpha:
            self.alpha.requires_grad = False
            self.alpha.weight = 0.
        if lesion_mu:
            self.mu.requires_grad = False
            self.mu.weight = 0.
        if lesion_gamma:
            self.gamma.requires_grad = False
            self.gamma.weight = 0.
        if lesion_kappa:
            self.kappa.requires_grad = False
            self.kappa.weight = 0.

    def forward(self, input_, inhibition, excitation,  activ=F.softplus, testmode=False):  # Worked with tanh and softplus
        # Attention gate: filter input_ and excitation
        # att_gate = torch.sigmoid(self.a_w_gate(input_) + self.a_u_gate(excitation))  # Attention Spotlight
        # att_gate = torch.sigmoid(self.a_w_gate(input_) * self.a_u_gate(excitation))  # Attention Spotlight
        if inhibition is None:
            inhibition = self.inh_init(input_)
        if excitation is None:
            excitation = self.exc_init(input_)
        if 1:  #  FORCING DEFAULTS FOR DEBUG  self.use_attention:
            # att_gate = torch.sigmoid(self.a_w_gate(inhibition) + self.a_u_gate(excitation))  # Attention Spotlight -- MOST RECENT WORKING
            att_gate = self.att_nl(self.gbn[0](self.a_w_gate(input_) + self.a_u_gate(excitation)))  # Attention Spotlight -- MOST RECENT WORKING
        elif not self.use_attention and testmode:
            att_gate = torch.zeros_like(input_)

        # Gate E/I with attention immediately
        if 1:  # self.use_attention:
            gated_input = input_  # * att_gate  # In activ range
            gated_excitation = att_gate * excitation  # att_gate * excitation
            gated_inhibition = inhibition
            # gated_inhibition = inhibition
        else:
            gated_input = input_
            gated_excitation = excitation
            gated_inhibition = inhibition

        if 1:  # not self.no_inh:
            # Compute inhibition
            # inh_intx = self.bn[0](F.conv2d(gated_excitation, self.w_inh, padding=self.h_padding))  # in activ range
            inh_intx = self.bn[0](self.w_inh(gated_excitation))  # , self.w_inh, padding=self.h_padding))  # in activ range
            inhibition_hat = activ(input_ - inh_intx * (self.alpha * gated_inhibition + self.mu))
            # inhibition_hat = activ(input_ - activ(inh_intx * (self.alpha * gated_inhibition + self.mu)))

            # Integrate inhibition
            inh_gate = torch.sigmoid(self.gbn[1](self.i_w_gate(gated_input) + self.i_u_gate(gated_inhibition)))
            inhibition = (1 - inh_gate) * inhibition + inh_gate * inhibition_hat  # In activ range
        else:
            inhibition, gated_inhibition = gated_excitation, excitation

        # Pass to excitatory neurons
        # exc_gate = torch.sigmoid(self.e_w_gate(inhibition) + self.e_u_gate(excitation))
        exc_gate = torch.sigmoid(self.gbn[2](self.e_w_gate(gated_inhibition) + self.e_u_gate(gated_excitation)))
        # exc_intx = self.bn[1](F.conv2d(inhibition, self.w_exc, padding=self.h_padding))  # In activ range
        exc_intx = self.bn[1](self.w_exc(inhibition))
        excitation_hat = activ(exc_intx * (self.kappa * inhibition + self.gamma))  # Skip connection OR add OR add by self-sim

        excitation = (1 - exc_gate) * excitation + exc_gate * excitation_hat
        if testmode:
            return inhibition, excitation, att_gate
        else:
            return inhibition, excitation


class FFhGRU(nn.Module):

    def __init__(self, dimensions, input_size=3, timesteps=8, kernel_size=15, jacobian_penalty=False, grad_method='bptt', no_inh=False, lesion_alpha=False, lesion_mu=False, lesion_gamma=False, lesion_kappa=False, nl=F.softplus, l1=0., output_size=2, num_rbp_steps=10, LCP=0., jv_penalty_weight=0.002):
        '''
        '''
        super(FFhGRU, self).__init__()
        self.timesteps = timesteps
        self.jacobian_penalty = jacobian_penalty
        self.grad_method = grad_method
        self.hgru_size = dimensions
        self.output_size = output_size
        self.num_rbp_steps = num_rbp_steps
        self.LCP = LCP
        self.jv_penalty_weight = jv_penalty_weight
        if l1 > 0:
            self.preproc = L1(nn.Conv2d(input_size, dimensions, kernel_size=7, stride=1, padding=7 // 2), weight_decay=l1)
        else:
            self.preproc = nn.Conv2d(input_size, dimensions, kernel_size=7, stride=1, padding=7 // 2)
        # self.preproc1 = nn.Conv2d(input_size, dimensions, kernel_size=7, stride=1, padding=7 // 2)
        self.unit1 = hConvGRUCell(
            input_size=input_size,
            hidden_size=self.hgru_size,
            kernel_size=kernel_size,
            use_attention=True,
            no_inh=no_inh,
            l1=l1,
            lesion_alpha=lesion_alpha,
            lesion_mu=lesion_mu,
            lesion_gamma=lesion_gamma,
            lesion_kappa=lesion_kappa,
            timesteps=timesteps)
        self.bn = nn.BatchNorm2d(self.hgru_size, eps=1e-03, affine=False, track_running_stats=True)
        # self.readout_bn = nn.BatchNorm2d(2, eps=1e-03, affine=True, track_running_stats=True)
        # self.readout_conv = nn.Conv2d(dimensions, 2, 1)
        # self.readout_dense = nn.Linear(2, 2)
        self.readout_bn = nn.BatchNorm2d(self.hgru_size, eps=1e-03, affine=True, track_running_stats=True)
        self.readout_dense = nn.Linear(self.hgru_size, self.output_size)
        self.nl = nl

    def forward(self, x, testmode=False):
        # First step: replicate x over the channel dim self.hgru_size times
        xbn = self.preproc(x)
        # xbn = self.bn(xbn)  # This might be hurting me...
        xbn = self.nl(xbn)  # TEST TO SEE IF THE NL STABLIZES

        # Extra preproc layer solves multi-color PF... Maybe channel pooling?
        # xbn = self.preproc1(x)
        # xbn = self.bn(xbn)  # This might be hurting me...
        # xbn = self.nl(xbn)  # TEST TO SEE IF THE NL STABLIZES

        # Now run RNN
        x_shape = xbn.shape
        excitation = None
        inhibition = None

        # Loop over frames
        states = []
        gates = []

        if self.grad_method == "bptt":
            for t in range(self.timesteps):
                out = self.unit1(
                    input_=xbn,
                    inhibition=inhibition,
                    excitation=excitation,
                    activ=self.nl,
                    testmode=testmode)
                if testmode:
                    inhibition, excitation, gate = out 
                    gates.append(gate)  # This should learn to keep the winner
                    states.append(self.readout_conv(excitation))  # This should learn to keep the winner
                else:
                    inhibition, excitation = out 
        elif self.grad_method == "rbp":
            with torch.no_grad():
               for t in range(self.timesteps - 1):
                    out = self.unit1(
                        input_=xbn,
                        inhibition=inhibition,
                        excitation=excitation,
                        activ=self.nl,
                        testmode=testmode)
                    if testmode:
                        inhibition, excitation, gate = out
                        gates.append(gate)  # This should learn to keep the winner
                        states.append(self.readout_conv(excitation))  # This should learn to keep the winner
                    else:
                        inhibition, excitation = out
            pen_exc = excitation.detach().requires_grad_()
            last_inh, last_exc = self.unit1(xbn, inhibition=inhibition, excitation=pen_exc, activ=self.nl, testmode=testmode)
            import pdb;pdb.set_trace()
            # Need to attach exc with inh to propoagate grads
            excitation = dummyhgru.apply(pen_exc, last_exc, self.num_rbp_steps)
        else:
            raise NotImplementedError(self.grad_method)

        # output = self.readout_bn(excitation)
        # output = self.readout_conv(output)
        # output = self.readout_conv(excitation)
        # output = excitation
        output = self.readout_bn(excitation)
        output = F.avg_pool2d(output, kernel_size=output.size()[2:])
        output = output.reshape(x_shape[0], -1)
        output = self.readout_dense(output)
        pen_type = 'l1'
        jv_penalty = torch.tensor([1]).float().cuda()
        if testmode: return output, torch.stack(states, 1), torch.stack(gates, 1)

        if self.training and self.LCP:
            norm_1_vect = torch.ones_like(last_exc)
            norm_1_vect.requires_grad = False
            jv_prod = torch.autograd.grad(last_exc, pen_exc, grad_outputs=[norm_1_vect], retain_graph=True, create_graph=True, allow_unused=True)[0]
            jv_penalty = ((jv_prod - self.LCP).clamp(0) ** 2).sum()
            out_dict = {"logits": output, "penalty": jv_penalty * self.jv_penalty_weight}
            return out_dict
        else:
            return output

