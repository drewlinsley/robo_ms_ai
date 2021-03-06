from torch import nn

from src.pl_modules import resnets


def get_network(name, num_classes):
    """Wrapper for selecting networks."""
    if name == "resnet18":
        net = resnets.resnet18(pretrained=False, num_classes=num_classes)
    elif name == "resnet18_pretrained_to_fc":
        net = resnets.resnet18(pretrained=True, num_classes=num_classes)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "resnet18_pretrained_to_fc_frozen":
        net = resnets.resnet18(pretrained=True)

        # Freeze the network
        for param in net.parameters():
            param.requires_grad = False

        # Train the FC
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "resnet18_pretrained_to_last_block_frozen":
        net = resnets.resnet18(pretrained=True)

        # Freeze the network
        for param in net.parameters():
            param.requires_grad = False

        # Train the FC
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "resnet50_pretrained_to_fc_frozen":
        net = resnets.resnet50(pretrained=True)

        # Freeze the network
        for param in net.parameters():
            param.requires_grad = False

        # Train the FC
        net.fc = nn.Linear(net.fc.in_features, num_classes)

    elif name == "resnet50_pretrained_to_last_block_frozen":
        net = resnets.resnet50(pretrained=True)

        # Freeze the network
        for name, param in net.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Train the FC
        import pdb;pdb.set_trace()
        net.fc = nn.Linear(net.fc.in_features, num_classes)

    elif name == "simclr_resnet18":
        net = resnets.resnet18(pretrained=False, num_classes=num_classes)
    else:
        raise NotImplementedError("Could not find network {}.".format(net))
    return net
