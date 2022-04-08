import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from captum.attr import visualization as viz
from heatmap import HeatMap
from skimage import io


#  non_nls = np.load("experiments/2022-03-27/17-53-42/attributions.npz", allow_pickle=True, encoding="latin1")
# nls = np.load("experiments/2022-03-27/18-35-18/attributions.npz", allow_pickle=True, encoding="latin1")

non_nls = np.load("experiments/2022-04-07/22-42-14/attributions.npz", allow_pickle=True, encoding="latin1")
# nls = np.load("experiments/2022-03-28/00-38-42/attributions.npz", allow_pickle=True, encoding="latin1")
nls = np.load("experiments/2022-04-07/22-42-29/attributions.npz", allow_pickle=True, encoding="latin1")


non_nls_imgs = non_nls["images"]
nls_imgs = nls["images"]

non_nls_attr = non_nls["viz"]
nls_attr = nls["viz"]

non_nls_pred = non_nls["pred"]
non_nls_gt = non_nls["gt"]

nls_pred = nls["pred"]
nls_gt = nls["gt"]

outlier_perc = 2
cmap = "Reds"

out_dir = os.path.join("data", "non_nls")
os.makedirs(out_dir, exist_ok=True)

# First go through and get the args + normalizers
max_val = 0
im_attrs = []
for idx, (im, attr, pred, gt) in enumerate(zip(non_nls_imgs, non_nls_attr, non_nls_pred, non_nls_gt)):
    # im, attr = im_attr
    im = im.transpose(1, 2, 0)
    im = (im - im.min()) / (im.max() - im.min())
    # im = im[..., [0]].repeat(3, -1)
    pred = np.argmax(pred)
    gt = gt[0]
    if pred == 0:
        pred = "dead"
        attr = attr[0]
    else:
        pred = "live"
        attr = attr[1]
    if gt == 0:
        gt = "dead"
    else:
        gt = "live"
    attr = np.maximum(attr, 0).mean(-1)
    attr = (attr - attr.min()) / (attr.max() - attr.min())  # max_val
    io.imsave(os.path.join(out_dir, "attr_{}_pred_{}_true_{}.png".format(idx, pred, gt)), (attr[..., None] * 255).astype(np.uint8))
    io.imsave(os.path.join(out_dir, "im_{}_pred_{}_true_{}.png".format(idx, pred, gt)), (im[..., [0]] * 255).astype(np.uint8))
    # hm = HeatMap(im, attr, gaussian_std=0, vmax=attr.max(), vmin=attr.min())
    f = plt.figure()
    plt.subplot(121)
    plt.imshow(im[100:200, 100:200])
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(im[100:200, 100:200], alpha=1)
    attr = np.ma.masked_where(attr <= 0.01, attr)
    attr = attr[100:200, 100:200]
    plt.imshow(attr, alpha=0.6, cmap=plt.cm.magma, interpolation="none")
    plt.axis("off")
    plt.colorbar()
    plt.suptitle("{}_pred_{}_true_{}.png".format(idx, pred, gt))
    plt.savefig(os.path.join(out_dir, "{}_pred_{}_true_{}.png".format(idx, pred, gt)))
    # hm.save(
    #     os.path.join(out_dir, "{}_pred_{}_true_{}".format(idx, pred, gt)),
    #     "png",
    #     transparency=0.3,
    #     color_map=cmap,
    #     show_axis=False,
    #     show_original=True,
    #     show_colorbar=True,
    #     width_pad=2)
    plt.close(f)

out_dir = os.path.join("data", "nls")
os.makedirs(out_dir, exist_ok=True)
for idx, (im, attr, pred, gt) in enumerate(zip(nls_imgs, nls_attr, nls_pred, nls_gt)):
    # im, attr = im_attr
    im = im.transpose(1, 2, 0)
    im = (im - im.min()) / (im.max() - im.min())
    # im = im[..., [0]].repeat(3, -1)
    pred = np.argmax(pred)
    gt = gt[0]
    if pred == 0:
        pred = "dead"
        attr = attr[0]
        attr = np.maximum(-attr, 0)
    else:
        pred = "live"
        attr = attr[1]
        attr = np.maximum(attr, 0)
    attr = np.maximum(attr, 0).mean(-1)
    attr = (attr - attr.min()) / (attr.max() - attr.min())  # max_val
    io.imsave(os.path.join(out_dir, "attr_{}_pred_{}_true_{}.png".format(idx, pred, gt)), (attr[..., None] * 255).astype(np.uint8))
    io.imsave(os.path.join(out_dir, "im_{}_pred_{}_true_{}.png".format(idx, pred, gt)), (im[..., [0]] * 255).astype(np.uint8))
    # attr0 = (attr[0] - attr[0].min()) / (attr[0].max() - attr[0].min())
    # attr1 = (attr[1] - attr[1].min()) / (attr[1].max() - attr[1].min())
    # attr = attr1 - attr0
    if gt == 0:
        gt = "dead"
    else:
        gt = "live"

    # hm = HeatMap(im, attr, gaussian_std=0, vmax=attr.max(), vmin=attr.min())
    f = plt.figure()
    plt.subplot(121)
    plt.imshow(im[100:200, 100:200])
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(im[100:200, 100:200], zorder=0, alpha=1)
    # attr = np.ma.masked_where(attr <= 0.01, attr)
    attr = attr[100:200, 100:200]
    plt.imshow(attr, alpha=0.5, cmap=plt.cm.RdBu_r, interpolation="none", zorder=1)
    plt.axis("off")
    plt.colorbar()
    plt.suptitle("{}_pred_{}_true_{}.png".format(idx, pred, gt))
    plt.savefig(os.path.join(out_dir, "{}_pred_{}_true_{}.png".format(idx, pred, gt)))
    # plt.show()
    # hm.save(
    #     os.path.join(out_dir, "{}_pred_{}_true_{}".format(idx, pred, gt)),
    #     "png",
    #     transparency=0.3,
    #     color_map=cmap,
    #     show_axis=False,
    #     show_original=True,
    #     show_colorbar=True,
    #     width_pad=2)
    plt.close(f)

