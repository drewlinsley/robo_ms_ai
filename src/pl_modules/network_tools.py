from torch import nn

from src.pl_modules import resnets


def get_network(name, num_classes):
	"""Wrapper for selecting networks."""
    if name == "resnet18":
        net = resnets.resnet18(pretrained=False, num_classes=num_classes)
    elif name == "resnet18_pretrained_to_fc":
        net = resnets.resnet18(pretrained=True, num_classes=num_classes)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "resnet18_pretrained_to_fc":
        net = resnets.resnet18(pretrained=True, num_classes=num_classes)

        # Freeze the network
        for param in net.parameters():
            param.requires_grad = False
        import pdb;pdb.set_trace()

        # Train the FC
        net.fc = nn.Linear(net.fc.in_features, num_classes)

    elif name == "simclr_resnet18":
        net = resnets.resnet18(pretrained=False, num_classes=num_classes)
    else:
        raise NotImplementedError("Could not find network {}.".format(net))
    return net
