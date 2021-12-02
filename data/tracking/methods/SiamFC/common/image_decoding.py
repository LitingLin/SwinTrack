import torchvision.io.image


class SiamFCImageDecodingProcessor:
    def __init__(self, post_processor=None):
        self.post_processor = post_processor

    def __call__(self, z_image_path, z_bbox, x_image_path, x_bbox, is_positive):
        z_image = torchvision.io.image.read_image(z_image_path, torchvision.io.image.ImageReadMode.RGB)
        if z_image_path != x_image_path:
            x_image = torchvision.io.image.read_image(x_image_path, torchvision.io.image.ImageReadMode.RGB)
        else:
            x_image = z_image
        data = (z_image, z_bbox, x_image, x_bbox, is_positive)
        if self.post_processor is not None:
            return self.post_processor(*data)
        else:
            return data
