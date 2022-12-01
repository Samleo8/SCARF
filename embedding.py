import torch

# TODO: Convert to NN module maybe?


def posemb_sincos_1d(N,
                     temperature=10000,
                     device=torch.device('cuda:0'),
                     dtype=torch.float32):
    '''
    Args:
        N: number of positions to embed
        temperature: Temperature of the positional encoding
        dtype: Data type of the positional encoding
    '''
    assert (N %
            2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    n = torch.arange(n, device=device)

    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1. / (temperature**omega)

    n = n.flatten()[:, None] * omega[None, :]
    pe = torch.cat((n.sin(), n.cos()), dim=1)
    return pe.type(dtype)


def posemb_sincos_2d(patches, temperature=10000, dtype=torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device=device),
                          torch.arange(w, device=device),
                          indexing='ij')
    assert (dim %
            4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
    omega = 1. / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)
