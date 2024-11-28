import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from torchvision import transforms

def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    return img.resize((new_w, new_h))

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

def crop_to_aspect_ratio(image, ratio="16:9"):
    width, height = image.size
    ratio_map = {
        "16:9": (16, 9),
        "4:3": (4, 3),
        "1:1": (1, 1)
    }
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        new_width = int(height * target_ratio_value)
        offset = (width - new_width) // 2
        crop_box = (offset, 0, offset + new_width, height)
    else:
        new_height = int(width / target_ratio_value)
        offset = (height - new_height) // 2
        crop_box = (0, offset, width, offset + new_height)

    cropped_img = image.crop(crop_box)
    return cropped_img


class PairCustomImageDataset(Dataset):
    def __init__(self, content_img_dir, style_img_dir, content_img_captions=None, style_img_captions=None, img_size=512, caption_type='json', random_ratio=False):

        self.content_images = [os.path.join(content_img_dir, i) for i in os.listdir(content_img_dir) if '.jpg' in i or '.png' in i]
        self.content_images.sort()

        self.style_images = [os.path.join(style_img_dir, i) for i in os.listdir(style_img_dir) if '.jpg' in i or '.png' in i]
        self.style_images.sort()

        assert len(self.content_images) == len(self.style_images) == 1, "Only one image per folder is currently supported."

        self.random_ratio = random_ratio

        self.img_size = img_size
        self.caption_type = caption_type
        if content_img_captions:
            self.content_img_captions = content_img_captions
        if style_img_captions:
            self.style_img_captions = style_img_captions

    def __len__(self):
        return len(self.content_images)

    def __getitem__(self, idx):
        try:

            content_img = Image.open(self.content_images[idx]).convert('RGB')
            style_img = Image.open(self.style_images[idx]).convert('RGB')

            if self.random_ratio:
                ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                if ratio != "default":
                    content_img = crop_to_aspect_ratio(content_img, ratio)
                    style_img = crop_to_aspect_ratio(style_img, ratio)
                if ratio == "default":
                    aspect_ratio = content_img.size[0] / content_img.size[1]
                elif ratio == "16:9":
                    aspect_ratio = 16. / 9.
                elif ratio == "4:3":
                    aspect_ratio = 4. / 3.
                else:
                    aspect_ratio = 1.0

            content_img = image_resize(content_img, self.img_size)
            w, h = content_img.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            content_img = content_img.resize((new_w, new_h))
            content_img = torch.from_numpy((np.array(content_img) / 127.5) - 1)
            content_img = content_img.permute(2, 0, 1)

            style_img = image_resize(style_img, self.img_size)
            w, h = style_img.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            style_img = style_img.resize((new_w, new_h))
            style_img = torch.from_numpy((np.array(style_img) / 127.5) - 1)
            style_img = style_img.permute(2, 0, 1)

            image_batch = torch.stack([content_img, style_img])

            image_transforms = transforms.Compose(
                [
                    transforms.RandomResizedCrop((new_h, new_w), scale=(.75, 1), ratio=(aspect_ratio, aspect_ratio)),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )

            image_batch = image_transforms(image_batch)
            content_img, style_img = image_batch[0], image_batch[1]

            content_json_path = self.content_images[idx].split('.')[0] + '.' + self.caption_type
            if self.content_img_captions:
                content_prompt = self.content_img_captions[idx]
            else:
                if self.caption_type == "json":
                    content_prompt = json.load(open(content_json_path))['caption']
                else:
                    content_prompt = open(content_json_path).read()

            style_json_path = self.content_images[idx].split('.')[0] + '.' + self.caption_type
            if self.style_img_captions:
                style_prompt = self.style_img_captions[idx]
            else:
                if self.caption_type == "json":
                    style_prompt = json.load(open(style_json_path))['caption']
                else:
                    style_prompt = open(style_json_path).read()

            return content_img, style_img, content_prompt, style_prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.content_images) - 1))

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_captions=None, img_size=512, caption_type='json', random_ratio=False):

        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()

        self.img_size = img_size
        self.caption_type = caption_type
        if img_captions:
            self.img_captions = img_captions
        self.random_ratio = random_ratio

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            if self.random_ratio:
                ratio = random.choice(["16:9", "default", "1:1", "4:3"])
                if ratio != "default":
                    img = crop_to_aspect_ratio(img, ratio)
            img = image_resize(img, self.img_size)
            w, h = img.size
            new_w = (w // 32) * 32
            new_h = (h // 32) * 32
            img = img.resize((new_w, new_h))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)


            json_path = self.images[idx].split('.')[0] + '.' + self.caption_type
            if self.img_captions:
                prompt = self.img_captions[idx]
            else:
                if self.caption_type == "json":
                    prompt = json.load(open(json_path))['caption']
                else:
                    prompt = open(json_path).read()
            return img, prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))

def loader(train_batch_size, num_workers, **args):
    dataset = PairCustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
    