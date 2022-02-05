import numpy as np
import collections
from typing import List, Any
import timeit
import attr
import torch
import torch.nn.init as init

from .helper import get_device, which_device
from .ode_models import refine


@attr.s(auto_attribs=True)
class Result:
    """A container class to collect training state"""
    model_list: List[Any]
    losses: Any
    refine_steps: Any
    train_acc: Any
    test_acc: Any
    epoch_times: List[Any]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def exp_lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, decayEpoch=[]):
    """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
    if epoch in decayEpoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay_rate
            print('lr decay update', param_group['lr'])
        return optimizer
    else:
        return optimizer


def reset_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print('reset lr', param_group['lr'])
    return optimizer


@torch.no_grad()
def calculate_accuracy(model, loader):
    device = which_device(model)
    correct, total_num = 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        total_num += len(data)
    return correct / total_num


def train_adapt(model,
                loader,
                testloader,
                criterion,
                N_epochs,
                N_refine=None,
                lr=1.0e-3,
                lr_decay=0.2,
                epoch_update=None,
                weight_decay=1e-5,
                refine_variance=0.0,
                device=None,
                fname=None,
                SAVE_DIR=None,
                criterion_per_el=None):
    """Adaptive Refinement Training for RefineNets"""
    if N_refine is None:
        N_refine = []
    if epoch_update is None:
        epoch_update = []
    if device is None:
        device = which_device(model)
    losses = []
    train_acc = []
    test_acc = []
    refine_steps = []
    epoch_times = []
    model_list = [model]
    lr_current = lr
    step_count = 0
    want_train_acc = False

    from libs.nero.optim.nero import Nero
    # optimizer = Nero(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # optimizer = torch.optim.RMSprop(
    #     model.parameters(),
    #     lr=lr,
    #     momentum=0.9,
    #     weight_decay=weight_decay,
    #     centered=True
    # )
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr, weight_decay=weight_decay)
    # Uncomment to use 4 gpus
    USE_PARALLEL = False
    if USE_PARALLEL:
        model_single = model
        model = torch.nn.DataParallel(model_single, device_ids=[0,1,2,3])
    # torch.backends.cudnn.flags(enabled=False)
    # torch.autograd.set_detect_anomaly(True)
    for e in range(N_epochs):
        model.train()

        # Make a new model if the epoch number is in the schedule
        # if e in N_refine:
        #     # Get back from parallel
        #     if USE_PARALLEL:
        #         model = model.module
        #     new_model = model.refine(refine_variance)
        #     model_list.append(new_model)
        #     # Make the new one parallel
        #     if USE_PARALLEL:
        #         model = torch.nn.DataParallel(new_model, device_ids=[0,1,2,3])
        #     else:
        #         model = new_model
        #     print('**** Allocated refinment ****')
        #     print('Total params: %.2fk' % (count_parameters(model)/1000.0))
        #     print('************')
        #     te_acc = calculate_accuracy(model, testloader)
        #     print('Test Accuracy after refinement: ', te_acc)
        #     test_acc.append( (e,te_acc) )
        #     if want_train_acc:
        #         tr_acc = calculate_accuracy(model, loader)
        #         print('Train Accuracy after refinement: ', tr_acc)
        #         train_acc.append( (e,tr_acc) )
        #     print(model)
        #     # We need to reset the optimizer to point to the new weights
        #     optimizer = torch.optim.SGD(model.parameters(), lr=lr_current, momentum=0.9, weight_decay=weight_decay)
        #     refine_steps.append(step_count)

        starting_time = timeit.default_timer()
        # Train one epoch over the new model
        model.train()
        iter_num = 0
        for imgs,labels in iter(loader):
            iter_num += 1
            imgs = imgs.to(device)
            labels = labels.to(device)
            # with torch.no_grad():
            out = model(imgs)

            # L = criterion(out,labels)
            L = 0.0
            #Lyapunov Regularization:
            yts = model.state_traj_output()
            # yts, yts_dot = model.full_traj_output()
            # yts.requires_grad = False
            # yts.requires_grad = False
            lya_truth = labels.expand(yts.shape[0], -1).flatten(0, 1)
            lya_state = yts.flatten(0, 1)
            v = criterion_per_el(lya_state, lya_truth).unflatten(0, yts.shape[:2])
            v_delta = v[1:] - v[:-1].detach()
            # v, vdot = torch.autograd.functional.jvp(func=lambda x: criterion_per_el(x, lya_truth),
            #                                      inputs=(lya_state,),
            #                                      v=yts_dot.flatten(0,1),
            #                                      create_graph=True)
            # v.detach_()
            # vdot.detach_()
            # vddot = torch.autograd.functional.jvp(
            #     func=lambda inputs: torch.autograd.functional.jvp(func=lambda\
            #         x: criterion_per_el(x, lya_truth),
            #                                      inputs=(inputs,),
            #                                      v=yts_dot.flatten(0,1),
            #                                      create_graph=True)[1],
            #     inputs=(lya_state,),
            #     v=yts_dot.flatten(0,1),
            #     create_graph=True)[1]
            violation = torch.relu(v_delta + 2.5* v[:-1].detach())
            # violation = torch.relu(vdot + 20 * v)
            # violation = torch.relu(vddot + 200 * vdot + 1000 * v)
            # yts.detach().cpu().numpy().astype('float32').tofile('yts_sample.dat')


            mean_violation = violation.mean()
            if iter_num == 0:
                print("[INFO]", end='')
            print(f"\rIteration: {iter_num} "
                  f"loss: {L:10.10f} "
                  f"violation mean: {mean_violation:10.10f} "
                  f"mean v: {v.mean():10.10f} "
                  f"mean vddelta: {v_delta.mean():10.10f}"
                  # f"mean vdot: {vdot.mean():10.10f} "
                  # f"mean vddot: {vddot.mean():10.10f}"
                  ,
                  end='',
                  flush=True)
            L = mean_violation
            # L = 0.1*violation.mean() + 100*L
            #lyapunov_training_ends
            L.backward()
            violation.grad=None
            violation=None
            mean_violation.grad=None
            mean_violation=None
            v=None
            # vdot.grad=None
            # vdot=None
            # yts.grad=None
            # yts_dot.grad=None
            yts=None
            yts_dot=None
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1000.)
            # print(f"[INFO] Iteration {iter_num}")
            # for name, param in model.named_parameters():
            #     if param is None:
            #         print(f"[PARAM] {name:37} : NONE")
            #         continue
            #     if param.grad is None:
            #         print(f"[PARAM] {name:37} "
            #               f"Mean Val: {param.abs().mean().item():5.5f} "
            #               f'Grad Val: None')
            #         continue
            #     print(f"[PARAM] {name:37} "
            #           f"Mean Val: {param.abs().mean().item():5.5f} "
            #           f"Mean Abs Grad: {param.grad.abs().mean().item():5.5f}")
            optimizer.step()
            optimizer.zero_grad()
            if torch.isnan(L):
                print("Hit a NaN, returning early.")
                return Result(model_list, losses, refine_steps, train_acc, test_acc, epoch_times)
            _loss = L.detach().cpu().item()
            losses.append(_loss)
            step_count+=1
        epoch_times.append(timeit.default_timer() - starting_time)
        #print("Epoch took ", epoch_times[-1], " seconds.")




        # Evaluate training and testing accuracy
        n_print = 5
        if e == 0 or (e+1) % n_print == 0:

            print('After Epoch: ', e+1)
            model.eval()
            if want_train_acc:
                tr_acc = calculate_accuracy(model, loader)
                print('Train Accuracy: ', tr_acc)
                train_acc.append( (e,tr_acc) )
            te_acc = calculate_accuracy(model, testloader)
            print(f'\n Test Accuracy: {te_acc} Test Error: {1- te_acc}')
            test_acc.append( (e,te_acc) )
        # Save checkpoint
        # if fname is not None and SAVE_DIR is not None and (e+1)%n_print==0:
        #     chckpt = Result(model_list, losses, refine_steps, train_acc, test_acc, epoch_times)
        #     try:
        #         os.mkdir(SAVE_DIR)
        #         print("Making directory ", SAVE_DIR)
        #     except:
        #         print("Directory ", SAVE_DIR, " already exists.")
        #     torch.save(chckpt, fname+f"-CHECKPOINT-{e}.pkl")

        # learnin rate schedule
        if e in epoch_update:
            lr_current *= lr_decay

        optimizer = exp_lr_scheduler(
            optimizer, e, lr_decay_rate=lr_decay, decayEpoch=epoch_update)


    print('Average time (without eval): ', np.mean(epoch_times))
    print('Total time (without eval): ', np.sum(epoch_times))
    return Result(model_list, losses, refine_steps, train_acc, test_acc, epoch_times)
