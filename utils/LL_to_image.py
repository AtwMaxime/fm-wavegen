from PIL import Image
import torch
import numpy as np

def rescale_to_255(tensor):
    """
    Rescales a tensor to the 0-255 range. This function is designed to work
    with 4-dimensional tensors as batched data, applying the rescaling
    recursively for each tensor along the batch dimension. The function
    normalizes the input tensor to the range [0, 1], then scales it to
    [0, 255], and clamps the values within the same range.

    :param tensor: Tensor to be rescaled. It can be either a single image
        tensor or a batch of image tensors. A 4-dimensional input tensor is
        treated as a batch, and the rescaling is applied to each element in
        the batch recursively.
    :return: A tensor rescaled to the range [0, 255].
    :rtype: torch.Tensor
    """
    if tensor.dim() == 4:
        return torch.stack([rescale_to_255(t) for t in tensor])

    t_min = tensor.min()
    t_max = tensor.max()
    scaled = (tensor - t_min) / (t_max - t_min + 1e-8)  # normalize to [0, 1]
    return (scaled * 255).clamp(0, 255)


def to_image(tensor, rescale=True):
    """
    Convert a tensor into an image or list of images. The function supports
    both batch tensors (4D) and single images (3D). When applied to a batch
    of images, the function recursively processes each image. The tensor
    must have its color channel as the first dimension. Optionally, the
    tensor can be rescaled to the range 0-255.

    :param tensor: The input tensor to convert. It can be 4D for a batch of
        images or 3D for a single image.
    :param rescale: A boolean indicating whether to rescale the tensor
        values to the range 0-255. Default is True.
    :return: An image if a single 3D tensor is provided or a list of images
        if a 4D batch tensor is given.
    """
    if tensor.dim() == 4:
        return [to_image(t, rescale) for t in tensor]

    assert tensor.shape[0] == 3
    arr = tensor.detach().cpu()

    if rescale:
        arr = rescale_to_255(arr)
    else:
        arr = arr.clamp(0, 255)

    img = Image.fromarray(arr.byte().numpy().transpose(1, 2, 0))
    return img

