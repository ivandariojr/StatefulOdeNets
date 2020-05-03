from typing import List
import attr
import driver


@attr.s(auto_attribs=True)
class Pack:
    dataset: str
    model: str
    scheme: str
    alpha: int
    initial_time_d: int  # t in [0,1] cut up into time_d slices
        # t=1, t=2, t=3,  -> if resnet epsilon = 1, then
        # t=1/3, t=2/3, t=1.0 -> resnet epsilon = 0.3
    time_epsilon: float  # xdot = epsilon * f(x,t)
    use_batch_norms: bool

    n_time_steps_per: int = 1
    epochs: int = 10
    # batch_size: int = 128
    # test_batch_size: int = 200

    lr: float = 0.05
    wd: float = 0
    use_adjoint: bool = True
    
    lr_decay: float = 0.1
    lr_update: List[int] = None
    refine: List[int] = None
    
args_list = [
    Pack("FMNIST", "SingleSegment", "rk4",       8, 3, 0.5, False),
    Pack("FMNIST", "SingleSegment", "midpoint",  8, 3, 0.5, False),
    Pack("FMNIST", "SingleSegment", "euler",     8, 3, 0.5, False),
    Pack("FMNIST", "SingleSegment", "rk4",      12, 3, 0.5, False),
    Pack("FMNIST", "SingleSegment", "midpoint", 12, 3, 0.5, False),
    Pack("FMNIST", "SingleSegment", "euler",    12, 3, 0.5, False),
    Pack("FMNIST", "SingleSegment", "rk4",      16, 3, 0.5, False),
    Pack("FMNIST", "SingleSegment", "midpoint", 16, 3, 0.5, False),
    Pack("FMNIST", "SingleSegment", "euler",    16, 3, 0.5, False),
]

for args in args_list:
    driver.do_a_train_set(
        args.dataset,
        args.model,
        args.alpha,
        args.scheme,
        args.use_batch_norms,
        args.initial_time_d,
        args.time_epsilon,
        args.n_time_steps_per,
        N_epochs=args.epochs,
        N_adapt=args.refine,
        lr=args.lr,
        lr_decay=args.lr_decay,
        epoch_update=args.lr_update,
        weight_decay=args.wd,
        use_adjoint=args.use_adjoint,
        device="cuda:0")