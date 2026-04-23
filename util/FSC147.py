from argparse import Namespace
import json
from pathlib import Path
import os

import numpy as np
import random
from torchvision import transforms
import torch
import cv2
import torchvision.transforms.functional as TF
import scipy.ndimage as ndimage
from PIL import Image
import argparse
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

MAX_HW = 384
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL')
    parser.add_argument('--mask_ratio', default=0.5, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true')
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N')
    parser.add_argument('--data_path', default='./data/FSC147/', type=str)
    parser.add_argument('--anno_file', default='annotation_FSC147_384.json', type=str)
    parser.add_argument('--data_split_file', default='Train_Test_Val_FSC_147.json', type=str)
    parser.add_argument('--im_dir', default='images_384_VarV2', type=str)
    parser.add_argument('--gt_dir', default='./data/FSC147/gt_density_map_adaptive_384_VarV2', type=str)
    parser.add_argument('--output_dir', default='./data/out/pre_4_dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='./weights/mae_pretrain_vit_base_full.pth')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--log_dir', default='./logs/pre_4_dir')
    parser.add_argument("--title", default="CounTR_pretraining", type=str)
    parser.add_argument("--wandb", default="counting", type=str)
    parser.add_argument("--team", default="wsense", type=str)
    parser.add_argument("--wandb_id", default=None, type=str)
    parser.add_argument("--do_aug", default=True, type=bool)
    parser.add_argument('--class_file', default='./data/FSC147/ImageClasses_FSC147.txt', type=str)
    return parser


def load_mask(mask_dir: str, im_id: str, box_idx: int, y1: int, x1: int, y2: int, x2: int, size=(64, 64)) -> torch.Tensor | None:
    mask_name = f"{im_id.split('.')[0]}.png" 
    mask_path = os.path.join(mask_dir, mask_name)
    if os.path.exists(mask_path):
        mask = Image.open(mask_path).convert('L')
        mask_tensor = transforms.ToTensor()(mask)
        mask_crop = mask_tensor[:, y1:y2 + 1, x1:x2 + 1]
        return transforms.Resize(size)(mask_crop)
    return None


def crop_and_resize_box(image_tensor: torch.Tensor,
                        y1: int, x1: int, y2: int, x2: int,
                        size=(64, 64)) -> torch.Tensor:
    """Crop a bounding box from image tensor and resize to target size."""
    bbox = image_tensor[:, y1:y2 + 1, x1:x2 + 1]
    return transforms.Resize(size)(bbox)


class ResizeSomeImage(object):
    """Base class: loads annotations, data split, and class dictionary."""

    def __init__(self, args):
        # Use the passed args directly (do NOT re-parse)
        self.data_path = Path(args.data_path)
        self.im_dir = self.data_path / args.im_dir

        with open(self.data_path / args.anno_file) as f:
            self.annotations = json.load(f)

        with open(self.data_path / args.data_split_file) as f:
            data_split = json.load(f)

        self.train_set = data_split['train']

        self.class_dict = {}
        if getattr(args, 'do_aug', False) and getattr(args, 'class_file', None):
            with open(self.data_path / args.class_file) as f:
                for line in f:
                    parts = line.split()
                    self.class_dict[parts[0]] = parts[1:]


class ResizePreTrainImage(ResizeSomeImage):
    """
    Resize image to 384×384 (divisible by 16), preserving aspect ratio.
    Used for MAE pre-training; density and box correctness not preserved.
    """

    def __init__(self, args, MAX_HW=384):
        super().__init__(args)
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image, lines_boxes, density = (
            sample['image'], sample['lines_boxes'], sample['gt_density']
        )

        W, H = image.size
        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)

        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_density = cv2.resize(density, (new_W, new_H))

        orig_count = np.sum(density)
        new_count = np.sum(resized_density)
        if new_count > 0:
            resized_density = resized_density * (orig_count / new_count)

        boxes = []
        for box in lines_boxes:
            y1, x1, y2, x2 = [int(k) for k in box]
            boxes.append([0, y1, x1, y2, x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = PreTrainNormalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)

        return {'image': resized_image, 'boxes': boxes, 'gt_density': resized_density}


class ResizeTrainImage(ResizeSomeImage):
    """
    Resize + augment for training.
    Reads exemplar boxes and applies segmentation masks (Masks_Pos / Masks_Neg)
    to remove background noise from the cropped bounding boxes.

    Augmentations: Gaussian noise, Color jitter, Gaussian blur,
                   Random affine, Random horizontal flip, Mosaic / Random crop.
    """

    def __init__(self, args, MAX_HW=384, do_aug=True):
        super().__init__(args)
        self.max_hw = MAX_HW
        self.do_aug = do_aug
        self.mask_pos_dir = os.path.join(args.data_path, "Masks_Pos")
        self.mask_neg_dir = os.path.join(args.data_path, "Masks_Neg")

    def __call__(self, sample):
        image, lines_boxes, neg_lines_boxes, dots, im_id, m_flag = (
            sample['image'], sample['lines_boxes'], sample['neg_lines_boxes'],
            sample['dots'], sample['id'], sample['m_flag']
        )

        W, H = image.size
        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)
        scale_factor_h = float(new_H) / H
        scale_factor_w = float(new_W) / W

        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_image = TTensor(resized_image)
        resized_density = np.zeros((new_H, new_W), dtype='float32')

        aug_flag = self.do_aug
        mosaic_flag = random.random() < 0.25

        if aug_flag:
            # --- Gaussian noise ---
            noise = torch.from_numpy(np.random.normal(0, 0.1, resized_image.size()))
            re_image = torch.clamp(resized_image + noise, 0, 1)

            # --- Color jitter + Gaussian blur ---
            re_image = Augmentation(re_image)

            # --- Random affine (with keypoint tracking) ---
            re1_image = re_image.transpose(0, 1).transpose(1, 2).numpy()
            keypoints = [
                Keypoint(
                    x=min(new_W - 1, int(dots[i][0] * scale_factor_w)),
                    y=min(new_H - 1, int(dots[i][1] * scale_factor_h))
                )
                for i in range(dots.shape[0])
            ]
            kps = KeypointsOnImage(keypoints, re1_image.shape)
            seq = iaa.Sequential([
                iaa.Affine(
                    rotate=(-15, 15),
                    scale=(0.8, 1.2),
                    shear=(-10, 10),
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
                )
            ])
            re1_image, kps_aug = seq(image=re1_image, keypoints=kps)

            # Build dot annotation map from augmented keypoints
            resized_density = np.zeros((new_H, new_W), dtype='float32')
            for i in range(len(kps.keypoints)):
                kp = kps_aug.keypoints[i]
                if (not kp.is_out_of_image(re1_image) and
                        int(kp.y) <= new_H - 1 and int(kp.x) <= new_W - 1):
                    resized_density[int(kp.y)][int(kp.x)] = 1
            resized_density = torch.from_numpy(resized_density)

            re_image = TTensor(re1_image)

            # --- Random horizontal flip ---
            if random.random() > 0.5:
                re_image = TF.hflip(re_image)
                resized_density = TF.hflip(resized_density)

            # --- Mosaic or random crop ---
            if mosaic_flag:
                reresized_image, reresized_density, m_flag = self._make_mosaic(
                    resized_image, dots, im_id,
                    new_H, new_W, scale_factor_h, scale_factor_w, m_flag
                )
            else:
                start = random.randint(0, new_W - 1 - 383)
                reresized_image = TF.crop(re_image, 0, start, 384, 384)
                reresized_density = resized_density[:, start:start + 384]

        else:
            # No augmentation: just place dots and crop
            for i in range(dots.shape[0]):
                resized_density[
                    min(new_H - 1, int(dots[i][1] * scale_factor_h))
                ][
                    min(new_W - 1, int(dots[i][0] * scale_factor_w))
                ] = 1
            resized_density = torch.from_numpy(resized_density)
            start = random.randint(0, new_W - self.max_hw)
            reresized_image = TF.crop(resized_image, 0, start, self.max_hw, self.max_hw)
            reresized_density = resized_density[0:self.max_hw, start:start + self.max_hw]

        # --- Gaussian density map + scale up ---
        reresized_density = ndimage.gaussian_filter(reresized_density.numpy(), sigma=(1, 1), order=0)
        reresized_density = torch.from_numpy(reresized_density * 60)

        # --- Crop exemplar boxes (positive), apply masks ---
        boxes, pos_rects = self._load_boxes(
            lines_boxes, resized_image, im_id,
            scale_factor_h, scale_factor_w, self.mask_pos_dir, max_boxes=3
        )

        # --- Crop negative exemplar boxes, apply masks ---
        neg_boxes, _ = self._load_boxes(
            neg_lines_boxes, resized_image, im_id,
            scale_factor_h, scale_factor_w, self.mask_neg_dir, max_boxes=3
        )
        IM_NORM_MEAN = [0.485, 0.456, 0.406]
        IM_NORM_STD = [0.229, 0.224, 0.225]
        normalize_fn = transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)

        reresized_image = normalize_fn(reresized_image) # Chuẩn hóa ảnh chính
        boxes = torch.stack([normalize_fn(b) for b in boxes]) 
        neg_boxes = torch.stack([normalize_fn(nb) for nb in neg_boxes])
        return {
            'image': reresized_image,
            'boxes': boxes,
            'neg_boxes': neg_boxes,
            'pos': pos_rects,
            'gt_density': reresized_density,
            'm_flag': m_flag
        }

    def _load_boxes(self, lines_boxes, resized_image, im_id,
                    scale_factor_h, scale_factor_w, mask_dir, max_boxes=3):
        """Crop, resize, and optionally mask-apply bounding boxes."""
        result_patches = []
        result_rects = []
        for cnt, box in enumerate(lines_boxes):
            if cnt >= max_boxes:
                break
            y1 = int(box[0] * scale_factor_h)
            x1 = int(box[1] * scale_factor_w)
            y2 = int(box[2] * scale_factor_h)
            x2 = int(box[3] * scale_factor_w)

            # Lưu tọa độ để vẽ lên WandB
            result_rects.append(torch.tensor([y1, x1, y2, x2]))

            bbox = crop_and_resize_box(resized_image, y1, x1, y2, x2, size=(64, 64))
            mask = load_mask(mask_dir, im_id, cnt, y1, x1, y2, x2, size=(64, 64))
            if mask is not None:
                bbox = bbox * mask
            result_patches.append(bbox)
            
        if len(result_patches) == 0:
            return torch.zeros(0, 3, 64, 64), torch.zeros(0, 4)
            
        return torch.stack(result_patches), torch.stack(result_rects)

    def _make_mosaic(self, resized_image, dots, im_id,
                     new_H, new_W, scale_factor_h, scale_factor_w, m_flag):
        """Build a 384×384 mosaic from 4 cropped image patches."""
        blending_l = random.randint(10, 20)
        resize_l = 192 + 2 * blending_l
        image_array, map_array = [], []

        if dots.shape[0] >= 70:
            # Dense scene: crop 4 patches from the same image
            for _ in range(4):
                length = random.randint(150, 384)
                start_W = random.randint(0, new_W - length)
                start_H = random.randint(0, new_H - length)
                patch = TF.crop(resized_image, start_H, start_W, length, length)
                patch = transforms.Resize((resize_l, resize_l))(patch)

                patch_density = np.zeros((resize_l, resize_l), dtype='float32')
                for i in range(dots.shape[0]):
                    dy = min(new_H - 1, int(dots[i][1] * scale_factor_h))
                    dx = min(new_W - 1, int(dots[i][0] * scale_factor_w))
                    if start_H <= dy < start_H + length and start_W <= dx < start_W + length:
                        ry = min(resize_l - 1, int((dy - start_H) * resize_l / length))
                        rx = min(resize_l - 1, int((dx - start_W) * resize_l / length))
                        patch_density[ry][rx] = 1

                image_array.append(patch)
                map_array.append(torch.from_numpy(patch_density))
        else:
            # Sparse scene: pick one real patch + 3 patches from other images
            m_flag = 1
            gt_pos = random.randint(0, 3) if random.random() > 0.25 else random.randint(0, 4)

            for i in range(4):
                if i == gt_pos:
                    Tim_id = im_id
                    Tdots = dots
                    r_image = resized_image
                    new_TH, new_TW = new_H, new_W
                    Tsfh, Tsfw = scale_factor_h, scale_factor_w
                else:
                    Tim_id = self.train_set[random.randint(0, len(self.train_set) - 1)]
                    Tdots = np.array(self.annotations[Tim_id]['points'])
                    Timage = Image.open('{}/{}'.format(self.im_dir, Tim_id))
                    Timage.load()
                    new_TW = 16 * int(Timage.size[0] / 16)
                    new_TH = 16 * int(Timage.size[1] / 16)
                    Tsfw = float(new_TW) / Timage.size[0]
                    Tsfh = float(new_TH) / Timage.size[1]
                    r_image = TTensor(transforms.Resize((new_TH, new_TW))(Timage))

                length = random.randint(250, 384)
                start_W = random.randint(0, new_TW - length)
                start_H = random.randint(0, new_TH - length)
                patch = TF.crop(r_image, start_H, start_W, length, length)
                patch = transforms.Resize((resize_l, resize_l))(patch)

                patch_density = np.zeros((resize_l, resize_l), dtype='float32')
                same_class = self.class_dict.get(im_id) == self.class_dict.get(Tim_id)
                if same_class:
                    for j in range(Tdots.shape[0]):
                        dy = min(new_TH - 1, int(Tdots[j][1] * Tsfh))
                        dx = min(new_TW - 1, int(Tdots[j][0] * Tsfw))
                        if start_H <= dy < start_H + length and start_W <= dx < start_W + length:
                            ry = min(resize_l - 1, int((dy - start_H) * resize_l / length))
                            rx = min(resize_l - 1, int((dx - start_W) * resize_l / length))
                            patch_density[ry][rx] = 1

                image_array.append(patch)
                map_array.append(torch.from_numpy(patch_density))

        # Blend 4 patches into a 384×384 mosaic
        def blend_horizontal(imgA, imgB, mapA, mapB):
            img = torch.cat((imgA[:, blending_l:resize_l - blending_l],
                             imgB[:, blending_l:resize_l - blending_l]), dim=1)
            dmap = torch.cat((mapA[blending_l:resize_l - blending_l],
                              mapB[blending_l:resize_l - blending_l]), dim=0)
            for k in range(blending_l):
                w1 = (blending_l - k) / (2 * blending_l)
                w2 = (k + blending_l) / (2 * blending_l)
                img[:, 192 + k] = imgA[:, resize_l - 1 - blending_l + k] * w1 + img[:, 192 + k] * w2
                img[:, 191 - k] = imgB[:, blending_l - k] * w1 + img[:, 191 - k] * w2
            return torch.clamp(img, 0, 1), dmap

        img5, map5 = blend_horizontal(image_array[0], image_array[1], map_array[0], map_array[1])
        img6, map6 = blend_horizontal(image_array[2], image_array[3], map_array[2], map_array[3])

        # Blend vertically
        out_img = torch.cat((img5[:, :, blending_l:resize_l - blending_l],
                             img6[:, :, blending_l:resize_l - blending_l]), dim=2)
        out_map = torch.cat((map5[:, blending_l:resize_l - blending_l],
                             map6[:, blending_l:resize_l - blending_l]), dim=1)
        for k in range(blending_l):
            w1 = (blending_l - k) / (2 * blending_l)
            w2 = (k + blending_l) / (2 * blending_l)
            out_img[:, :, 192 + k] = img5[:, :, resize_l - 1 - blending_l + k] * w1 + out_img[:, :, 192 + k] * w2
            out_img[:, :, 191 - k] = img6[:, :, blending_l - k] * w1 + out_img[:, :, 191 - k] * w2
        out_img = torch.clamp(out_img, 0, 1)

        return out_img, out_map, m_flag


class ResizeValImage(ResizeSomeImage):
    """
    Resize validation image to MAX_HW × MAX_HW.
    Reads exemplar boxes and applies segmentation masks (Masks_Pos / Masks_Neg).
    """

    def __init__(self, args, MAX_HW=384):
        super().__init__(args)
        self.max_hw = MAX_HW
        self.mask_pos_dir = os.path.join(args.data_path, "Masks_Pos")
        self.mask_neg_dir = os.path.join(args.data_path, "Masks_Neg")

    def __call__(self, sample):
        image, dots, m_flag, lines_boxes, neg_lines_boxes, im_id = (
            sample['image'], sample['dots'], sample['m_flag'],
            sample['lines_boxes'], sample['neg_lines_boxes'], sample['id']
        )

        W, H = image.size
        new_H = new_W = self.max_hw
        scale_factor_h = float(new_H) / H
        scale_factor_w = float(new_W) / W

        resized_image = TTensor(transforms.Resize((new_H, new_W))(image))

        # Build Gaussian density map
        resized_density = np.zeros((new_H, new_W), dtype='float32')
        for i in range(dots.shape[0]):
            resized_density[
                min(new_H - 1, int(dots[i][1] * scale_factor_h))
            ][
                min(new_W - 1, int(dots[i][0] * scale_factor_w))
            ] = 1
        resized_density = torch.from_numpy(
            ndimage.gaussian_filter(resized_density, sigma=4, order=0)
        ) * 60

        # --- Positive exemplar boxes ---
        boxes, rects = [], []
        for cnt, box in enumerate(lines_boxes):
            if cnt >= 3:
                break
            y1 = int(box[0] * scale_factor_h)
            x1 = int(box[1] * scale_factor_w)
            y2 = int(box[2] * scale_factor_h)
            x2 = int(box[3] * scale_factor_w)
            rects.append(torch.tensor([y1, x1, y2, x2]))

            bbox = crop_and_resize_box(resized_image, y1, x1, y2, x2, size=(64, 64))
            mask = load_mask(self.mask_pos_dir, im_id, cnt, size=(64, 64))
            if mask is not None:
                bbox = bbox * mask
            boxes.append(bbox)

        # --- Negative exemplar boxes ---
        neg_boxes, neg_rects = [], []
        for cnt, box in enumerate(neg_lines_boxes):
            if cnt >= 3:
                break
            y1 = int(box[0] * scale_factor_h)
            x1 = int(box[1] * scale_factor_w)
            y2 = int(box[2] * scale_factor_h)
            x2 = int(box[3] * scale_factor_w)
            neg_rects.append(torch.tensor([y1, x1, y2, x2]))

            neg_bbox = crop_and_resize_box(resized_image, y1, x1, y2, x2, size=(64, 64))
            mask = load_mask(self.mask_neg_dir, im_id, cnt, size=(64, 64))
            if mask is not None:
                neg_bbox = neg_bbox * mask
            neg_boxes.append(neg_bbox)

        return {
            'image': resized_image,
            'boxes': torch.stack(boxes),
            'neg_boxes': torch.stack(neg_boxes),
            'pos': torch.stack(rects),
            'gt_density': resized_density,
            'm_flag': m_flag
        }




PreTrainNormalize = transforms.Compose([
    transforms.RandomResizedCrop(MAX_HW, scale=(0.2, 1.0), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

TTensor = transforms.Compose([transforms.ToTensor()])

Augmentation = transforms.Compose([
    transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.15, hue=0.15),
    transforms.GaussianBlur(kernel_size=(7, 9))
])

Normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
])


def transform_train(args: Namespace, do_aug=True):
    return transforms.Compose([ResizeTrainImage(args, MAX_HW, do_aug)])

def transform_val(args: Namespace):
    return transforms.Compose([ResizeValImage(args, MAX_HW)])

def transform_pre_train(args: Namespace):
    return transforms.Compose([ResizePreTrainImage(args, MAX_HW)])