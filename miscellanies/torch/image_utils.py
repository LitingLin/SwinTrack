import torch
import torchvision


def decode_image_as_torch_tensor(file_path):
    image = torchvision.io.image.read_image(file_path, torchvision.io.image.ImageReadMode.RGB)
    image = image.to(torch.float)
    image /= 255.
    return image


def get_pil_image_from_torch_tensor(torch_tensor: torch.Tensor):
    return torchvision.transforms.ToPILImage()(torch_tensor).convert("RGB")
