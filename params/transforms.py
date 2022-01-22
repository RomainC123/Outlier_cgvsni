import torchvision.transforms as transforms

CROP_SIZE = 233

TRANSFORMS_TRAIN = transforms.Compose([transforms.CenterCrop(CROP_SIZE),
                                      transforms.ToTensor()])
