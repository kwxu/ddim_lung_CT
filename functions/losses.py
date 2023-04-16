import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          x_cond: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor,
                          mask=None,
                          keepdim=False,
                          loss_type='l2',
                          restricted_loss_region=False
                          ):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float(), x_cond, mask)

    if loss_type == 'l2':
        loss = (e - output).square()
    elif loss_type == 'l1':
        loss = (e - output).abs()
    else:
        raise NotImplementedError

    if restricted_loss_region:
        loss[mask == 1] = 0

    loss = loss.sum(dim=(1, 2, 3))

    if keepdim:
        return loss
    else:
        return loss.mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
