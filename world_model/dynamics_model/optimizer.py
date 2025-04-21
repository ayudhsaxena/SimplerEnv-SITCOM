from torch.optim import Adam, AdamW


def separate_weight_decayable_params(params, if_named_params=False):
    wd_params, no_wd_params = [], []
    for param in params:
        if if_named_params:
            name, param = param
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


def get_optimizer(
    params, lr=1e-4, wd=1e-2, betas=(0.9, 0.99), eps=1e-8, filter_by_requires_grad=False, group_wd_params=True, **kwargs
):
    if filter_by_requires_grad:
        params = list(filter(lambda t: t[1].requires_grad, params))

    if wd == 0:
        params = [param for _, param in params]
        return Adam(params, lr=lr, betas=betas, eps=eps)

    if group_wd_params:
        wd_params, no_wd_params = separate_weight_decayable_params(params, if_named_params=True)

        params = [
            {"params": wd_params},
            {"params": no_wd_params, "weight_decay": 0},
        ]

    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)


def separate_pretrain_params(named_params, pretrain_filter_key):
    pretrain_params, no_pretrain_params = [], []
    p_name, np_names = [], []
    for name, param in named_params:
        param_list = pretrain_params if pretrain_filter_key in name else no_pretrain_params
        name_list = p_name if pretrain_filter_key in name else np_names
        param_list.append(param)
        name_list.append(name)
    print("=================================")
    print(f"Pretrain params: {p_name}")
    print(f"No pretrain params: {np_names}")
    print("=================================")
    return pretrain_params, no_pretrain_params


def get_optimizer_pretrained_init(
    named_params,
    pretrain_filter_key,
    lr=1e-4,
    pretrain_lr_mult=0.01,
    wd=1e-2,
    pretrain_wd_mult=0.01,
    betas=(0.9, 0.99),
    eps=1e-8,
    filter_by_requires_grad=False,
    group_wd_params=True,
    **kwargs,
):
    if filter_by_requires_grad:
        named_params = list(filter(lambda t: t[1].requires_grad, named_params))

    pretrain_params, no_pretrain_params = separate_pretrain_params(named_params, pretrain_filter_key)
    if wd == 0:
        return Adam(
            [{"params": no_pretrain_params}, {"params": pretrain_params, "lr": lr * pretrain_lr_mult}],
            lr=lr,
            betas=betas,
            eps=eps,
        )

    if group_wd_params:
        wd_pretrain_params, no_wd_pretrain_params = separate_weight_decayable_params(pretrain_params)
        wd_no_pretrain_params, no_wd_no_pretrain_params = separate_weight_decayable_params(no_pretrain_params)
        params = [
            {"params": wd_pretrain_params, "weight_decay": wd * pretrain_wd_mult, "lr": lr * pretrain_lr_mult},
            {"params": no_wd_pretrain_params, "weight_decay": 0, "lr": lr * pretrain_lr_mult},
            {"params": wd_no_pretrain_params},
            {"params": no_wd_no_pretrain_params, "weight_decay": 0},
        ]

    return AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
