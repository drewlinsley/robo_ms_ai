from warnings import warn
try:
    from torchvision import transforms
except ImportError:
    warn('You want to use `torchvision` which is not installed yet,'  # pragma: no-cover
         ' install it with `pip install torchvision`.')



def COR14_normalization(maxval=33000):
    normalize = transforms.Normalize(mean=[x / maxval for x in [1086.6762200888888]],
                                     std=[x / maxval for x in [2019.9389348809887]])
    return normalize


def NuclearGedi_normalization(maxval=2 ** 16):
    # First normalize to [0, 1]
    gedi_mean = [0]
    gedi_std = [maxval]
    gedi_normalize = transforms.Normalize(
        mean=gedi_mean,
        std=gedi_std)

    # Then normalize to imagenet for pretrained models
    in_mean = [0.485, 0.456, 0.406]
    in_std = [0.229, 0.224, 0.225]
    in_normalize = transforms.Normalize(
        mean=in_mean,
        std=in_std)

    # Now compose the normalizations
    normalize = transforms.Compose([gedi_normalize, in_normalize])
    return normalize


def cifar10_normalization():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    return normalize
