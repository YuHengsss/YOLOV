import torch
import torch.nn.functional as F
import torch.nn as nn



def index_sample_2d(tensor, index):
    assert tensor.ndim == 2
    assert index.ndim == 2
    assert index.dtype == torch.int64
    d0, d1 = tensor.shape
    d2, d3 = index.shape
    assert d0 == d2
    tensor_ = tensor.reshape((-1, ))
    batch_ind = torch.arange(end=d0, dtype=index.dtype).unsqueeze(-1) * d1
    batch_ind = batch_ind.to(index.device)
    index_ = index + batch_ind
    index_ = index_.reshape((-1, ))
    out = tensor_[index_]
    out = out.reshape((d2, d3))
    return out


def gather_1d(tensor, index):
    assert index.ndim == 1
    assert index.dtype == torch.int64
    # d0, d1 = tensor.shape
    # d2, d3 = index.shape
    # assert d0 == d2
    # tensor_ = tensor.reshape((-1, ))
    # batch_ind = torch.arange(end=d0, dtype=index.dtype).unsqueeze(-1) * d1
    # index_ = index + batch_ind
    # index_ = index_.reshape((-1, ))
    # out = tensor_[index_]
    out = tensor[index]
    return out



def gather_nd(tensor, index):
    if tensor.ndim == 4 and index.ndim == 2:
        N, R, S, T = tensor.shape
        index_0 = index[:, 0]  # [M, ]
        index_1 = index[:, 1]  # [M, ]
        index_2 = index[:, 2]  # [M, ]
        index_ = index_0 * R * S + index_1 * S + index_2  # [M, ]
        x2 = torch.reshape(tensor, (N * R * S, T))  # [N*R*S, T]
        index_ = index_.to(torch.int64)
        out = gather_1d(x2, index_)
    elif tensor.ndim == 3 and index.ndim == 3:
        A, B, C = tensor.shape
        D, E, F = index.shape
        assert F == 2
        # out.shape = [D, E, C]
        tensor_ = tensor.reshape((-1, C))   # [A*B, C]
        index_ = index.reshape((-1, F))     # [D*E, F]


        index_0 = index_[:, 0]  # [D*E, ]
        index_1 = index_[:, 1]  # [D*E, ]
        index_ = index_0 * B + index_1  # [D*E, ]

        out = gather_1d(tensor_, index_)  # [D*E, C]
        out = out.reshape((D, E, C))   # [D, E, C]
    else:
        raise NotImplementedError("not implemented.")
    return out


def identity(x):
    return x


def mish(x):
    return F.mish(x) if hasattr(F, mish) else x * F.tanh(F.softplus(x))


def swish(x):
    return x * torch.sigmoid(x)

TRT_ACT_SPEC = {'swish': swish}

ACT_SPEC = {'mish': mish, 'swish': swish}


def get_act_fn(act=None, trt=False):
    assert act is None or isinstance(act, (
        str, dict)), 'name of activation should be str, dict or None'
    if not act:
        return identity

    if isinstance(act, dict):
        name = act['name']
        act.pop('name')
        kwargs = act
    else:
        name = act
        kwargs = dict()

    if trt and name in TRT_ACT_SPEC:
        fn = TRT_ACT_SPEC[name]
    elif name in ACT_SPEC:
        fn = ACT_SPEC[name]
    else:
        fn = getattr(F, name)

    return lambda x: fn(x, **kwargs)


def get_static_shape(tensor):
    # shape = torch.shape(tensor)
    # shape.requires_grad_(False)
    # return shape
    return tensor.shape


def paddle_distributed_is_initialized():
    return True


