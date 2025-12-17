import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import LBFGS
import os
import argparse
from models.definitions.vgg19 import Vgg19

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


# ================= IMAGE UTILS (UNCHANGED) =================

def load_image(img_path, target_shape="None"):
    if not os.path.exists(img_path):
        raise Exception(f'Path not found: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]
    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            h, w = img.shape[:2]
            new_h = target_shape
            new_w = int(w * (new_h / h))
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_CUBIC)
        else:
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)
    img = img.astype(np.float32)
    img /= 255.0
    return img


def prepare_img(img_path, target_shape, device):
    img = load_image(img_path, target_shape)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])
    return transform(img).to(device).unsqueeze(0)


def save_and_maybe_display(optimizing_img, dump_path, config, img_id, num_of_iterations):
    if img_id != num_of_iterations - 1:
        return
    out_img = optimizing_img.squeeze(0).cpu().detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)
    out_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
    out_img = np.clip(out_img, 0, 255).astype('uint8')
    cv.imwrite(os.path.join(dump_path, "output_image.png"), out_img[:, :, ::-1])


# ================= MODEL / LOSS (UNCHANGED) =================

def prepare_model(device):
    model = Vgg19(requires_grad=False, show_progress=True)
    return model.to(device).eval(), model.content_feature_maps_index, model.style_feature_maps_indices


def gram_matrix(x, normalize=True):
    b, ch, h, w = x.size()
    features = x.view(b, ch, w * h)
    gram = features.bmm(features.transpose(1, 2))
    return gram / (ch * h * w) if normalize else gram


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))


def build_loss(net, img, targets, c_idx, s_idxs, cfg):
    tgt_c, tgt_s = targets
    cur = net(img)

    content_loss = torch.nn.MSELoss()(tgt_c, cur[c_idx].squeeze(0))

    style_loss = 0
    for g1, g2 in zip(tgt_s, [gram_matrix(cur[i]) for i in s_idxs]):
        style_loss += torch.nn.MSELoss(reduction='sum')(g1[0], g2[0])
    style_loss /= len(tgt_s)

    tv_loss = total_variation(img)

    total = (
        cfg['content_weight'] * content_loss +
        cfg['style_weight'] * style_loss +
        cfg['tv_weight'] * tv_loss
    )

    return total, content_loss, style_loss, tv_loss


# ================= NST CORE (UNCHANGED) =================

def neural_style_transfer(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content = prepare_img(os.path.join(cfg['run_dir'], "content_image.png"), cfg['height'], device)
    style = prepare_img(os.path.join(cfg['run_dir'], "style_image.png"), cfg['height'], device)

    img = Variable(content.clone(), requires_grad=True)

    net, c_idx, s_idxs = prepare_model(device)

    tgt_c = net(content)[c_idx].squeeze(0)
    tgt_s = [gram_matrix(x) for i, x in enumerate(net(style)) if i in s_idxs]

    targets = [tgt_c, tgt_s]

    optimizer = LBFGS((img,), max_iter=cfg['num_of_iterations'], line_search_fn='strong_wolfe')
    cnt = 0

    def closure():
        nonlocal cnt
        optimizer.zero_grad()
        total, c, s, tv = build_loss(net, img, targets, c_idx, s_idxs, cfg)
        total.backward()
        with torch.no_grad():
            print(
                f"Iteration {cnt:03d} | "
                f"Total Loss: {total.item():.4f} | "
                f"Content: {c.item():.4f} | "
                f"Style: {s.item():.4f} | "
                f"TV: {tv.item():.4f}"
            )
            save_and_maybe_display(img, cfg['run_dir'], cfg, cnt, cfg['num_of_iterations'])
        cnt += 1
        return total

    optimizer.step(closure)


# ================= ARGUMENT PARSING (ALLOWED) =================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--content_weight", type=float, default=100000.0)
    parser.add_argument("--style_weight", type=float, default=30000.0)
    parser.add_argument("--tv_weight", type=float, default=1.0)
    parser.add_argument("--num_of_iterations", type=int, default=1000)

    args = parser.parse_args()

    config = vars(args)
    neural_style_transfer(config)
