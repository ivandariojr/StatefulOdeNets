import torch
from torch import nn
import math
from .ode_models import *


def NoSequential(*args):
    """Filters Nones as no-ops when making a nn.Sequential to allow for architecture toggling."""
    net = [ arg for arg in args if arg is not None ]
    return nn.Sequential(*net)


def compose_list(func_list, x):
    output = x
    for func in func_list:
        output = func(output)
    return output

class ContinuousNet(nn.Module):
    """Uses one OdeBlock per segment."""
    def __init__(self,
                 ALPHA=16,
                 scheme='euler',
                 time_d=1,
                 n_in_channels=3,
                 n_outputs=10,
                 use_batch_norms=True,
                 time_epsilon=1.0,
                 n_time_steps_per=1,
                 use_skip_init=False,
                 use_stitch=True,
                 use_adjoint=False,
                 activation_before_conv=False):
        super().__init__()
        self.scheme = scheme
        self.time_d = time_d
        self.use_batch_norms = use_batch_norms
        self.stitch_epsilon = time_epsilon / (time_d*n_time_steps_per)

        if activation_before_conv:
            _OdeUnit = ShallowConv2DODE_Flipped
            _ODEStitch = ODEStitch_Flipped
        else:
            _OdeUnit = ShallowConv2DODE
            _ODEStitch = ODEStitch

        # This macro lets us make 3 of them concisely without typos
        _macro = lambda _alpha : \
            ODEBlock(
                _OdeUnit(
                    time_d,
                    _alpha,
                    _alpha,
                    epsilon=time_epsilon,
                    use_batch_norms=use_batch_norms,
                    use_skip_init=use_skip_init),
                n_time_steps=time_d*n_time_steps_per,
                scheme=scheme,
                use_adjoint=use_adjoint)
        if use_stitch:
            _stitch_macro = lambda _alpha, _beta, stride=2 : \
                _ODEStitch(_alpha, _beta, _beta,
                          epsilon=self.stitch_epsilon,
                          use_batch_norms=use_batch_norms,
                          use_skip_init=use_skip_init,
                          stride=stride)
        else:
            _stitch_macro = lambda _alpha, _beta : \
                nn.Conv2d(_alpha, _beta, kernel_size=1, padding=1, stride=2, bias=False)

        self.dyns = nn.ModuleList([
            _macro(ALPHA),
            _macro(2 * ALPHA),
            _macro(4 * ALPHA)
        ])
        self.stiches = nn.ModuleList([
            _stitch_macro(ALPHA, 2 * ALPHA),
            _stitch_macro(2 * ALPHA, 4 * ALPHA)
        ])
        self.output = NoSequential(
            nn.BatchNorm2d(4 * ALPHA, momentum=0.9) if activation_before_conv else None,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4 * ALPHA, n_outputs)
        )
        # The full network, with three OdeBlocks (_macro)
        self.net = NoSequential(
            nn.Conv2d(
                n_in_channels, ALPHA, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ALPHA) if use_batch_norms else None,
            nn.ReLU(),
            _stitch_macro(ALPHA, ALPHA, stride=1),
            self.dyns[0],
            self.stiches[0],
            self.dyns[1],
            self.stiches[1],
            self.dyns[2],
            self.output
        )

        flat_module_list = list()
        for m in self.modules():
            if isinstance(m, list):
                flat_module_list += m
            else:
                flat_module_list += [m]

        for m in flat_module_list:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Conv2DODE):
                n = m.width * m.width * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        return self.net(x)

    def full_traj_output(self):
        outs = []
        outs_dot = []
        for i, dyn in enumerate(self.dyns):
            orig_shape = (dyn.h.shape[0], dyn.h.shape[1])
            h = dyn.h.flatten(0, 1)
            out_coord_list = []
            for j in range(i, len(self.stiches)):
                out_coord_list += [self.stiches[j]]
            out_coord_list += [self.output]
            out_i, jvp = torch.autograd.functional.jvp(
                func=lambda x: compose_list(out_coord_list, x),
                inputs=(h,),
                v=dyn(h),
                create_graph=True
            )

            outs.append(out_i.unflatten(0, orig_shape))
            outs_dot.append(jvp.unflatten(0, orig_shape))
        return torch.cat(outs, dim=0), torch.cat(outs_dot, dim=0)

    def state_traj_output(self):
        outs = []
        for i, dyn in enumerate(self.dyns):
            if i == 0:
                orig_shape = (dyn.h.shape[0], dyn.h.shape[1])
                h = dyn.h.flatten(0, 1)
            else:
                orig_shape = (dyn.h.shape[0]-1, dyn.h.shape[1])
                h = dyn.h[1:].flatten(0, 1)
            out_coord_list = []
            for j in range(i, len(self.stiches)):
                out_coord_list += [self.stiches[j]]
            out_coord_list += [self.output]
            out_i = compose_list(out_coord_list, h)
            outs.append(out_i.unflatten(0, orig_shape))
        return torch.cat(outs, dim=0)

    def refine(self, variance=0.0):
        new = copy.deepcopy(self)
        new.time_d = 2*self.time_d
        new.scheme = self.scheme
        new.net = nn.Sequential(*[ refine(mod, variance) for mod in self.net])
        return new
