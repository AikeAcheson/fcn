import torch
from model import get_FCN
from Utils import data
from Utils import train
import torchvision
import matplotlib.pyplot as plt
from torch import nn

# Load model.
num_classes = 21
net = get_FCN(num_classes)
devices = train.try_all_gpus()
net = nn.DataParallel(net, device_ids=devices).to(devices[0])
net.load_state_dict(torch.load('checkpoints/fcn19.pt'))
net.eval()

# Load data.
batch_size, crop_size = 32, (320, 480)
train_iter, test_iter = data.load_data_voc(batch_size, crop_size)


def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])


def label2image(pred):
    colormap = torch.tensor(data.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]


# Load data.
voc_dir = data.download_extract('voc2012', 'VOCdevkit/VOC2012')
test_images, test_labels = data.read_voc_images(voc_dir, False)
n, imgs = 4, []

# Predict.
for i in [1, 7, 9, 11]:
    crop_rect = (0, 0, 320, 480)
    X = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    pred = label2image(predict(X))
    imgs += [X.permute(1, 2, 0),
             pred.cpu(),
             torchvision.transforms.functional.crop(test_labels[i], *crop_rect).permute(1, 2, 0)]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
plt.show()
