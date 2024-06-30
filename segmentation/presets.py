import torch
import torchvision.transforms.v2 as T
import torchvision.tv_tensors as tv_tensors
import v2_extras


class SegmentationPresetTrain:
    def __init__(self, *, base_size, crop_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        self.transforms = T.Compose([
            T.RandomResize(min_size=int(0.5 * base_size), max_size=int(2.0 * base_size)),
            T.RandomHorizontalFlip(),
            v2_extras.PadIfSmaller(crop_size, fill={tv_tensors.Mask: 255, "others": 0}),
            T.RandomCrop(crop_size),
            T.PILToTensor(),
            T.ToDtype(dtype={torch.Tensor: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True),
            T.Normalize(mean=mean, std=std),
            T.ToPureTensor()
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, *, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        self.transforms = T.Compose([
            T.Resize(size=(base_size, base_size)),
            T.ToImage(),
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
            T.ToPureTensor()
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)
