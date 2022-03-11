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


def cifar10_normalization():
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    return normalize
