import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import glob
import random

import torchvision.transforms.functional as TF
from torch.distributions import Normal
import torch
import numpy as np
import torch
from basicsr.utils.registry import DATASET_REGISTRY
import json # New import for JSON loading
import os.path as osp # New import for path manipulation
# TODO: update it with exdark

class RandomGammaCorrection(object):
    def __init__(self, gamma = None):
        self.gamma = gamma
    def __call__(self,image):
        if self.gamma == None:
            # more chances of selecting 0 (original image)
            gammas = [0.5,1,2]
            self.gamma = random.choice(gammas)
            return TF.adjust_gamma(image, self.gamma, gain=1)
        elif isinstance(self.gamma,tuple):
            gamma=random.uniform(*self.gamma)
            return TF.adjust_gamma(image, gamma, gain=1)
        elif self.gamma == 0:
            return image
        else:
            return TF.adjust_gamma(image,self.gamma,gain=1)

def remove_background(image_tensor):
    # 输入应为 PyTorch Tensor (C, H, W)
    # 将 Tensor 转换为 NumPy 数组 (H, W, C) 进行操作
    image_np = image_tensor.cpu().numpy().transpose(1, 2, 0)
    _EPS = 1e-7
    rgb_max = np.max(image_np, (0, 1))
    rgb_min = np.min(image_np, (0, 1))
    
    # 避免除以零，确保 rgb_max - rgb_min 不为零
    denominator = rgb_max - rgb_min
    denominator[denominator == 0] = _EPS # 防止除以零

    image_processed_np = (image_np - rgb_min) * rgb_max / (denominator + _EPS) # 添加 _EPS 再次确保安全
    
    # 将处理后的 NumPy 数组 (H, W, C) 转回 PyTorch Tensor (C, H, W)
    image_processed_tensor = torch.from_numpy(image_processed_np.transpose(2, 0, 1)).float()
    return image_processed_tensor

def glod_from_folder(folder_list, index_list):
    ext = ['png','jpeg','jpg','bmp','tif']
    index_dict={}
    for i,folder_name in enumerate(folder_list):
        data_list=[]
        [data_list.extend(glob.glob(folder_name + '/*.' + e)) for e in ext]
        data_list.sort()
        index_dict[index_list[i]]=data_list
    return index_dict

class Flare_Image_Loader(data.Dataset):
    def __init__(self, image_path ,transform_base,transform_flare,mask_type=None):
        self.ext = ['png','jpeg','jpg','bmp','tif']
        self.data_list=[]
        [self.data_list.extend(glob.glob(image_path + '/*.' + e)) for e in self.ext]
        self.data_list.sort() # 确保数据列表排序一致

        self.flare_dict={}
        self.flare_list=[]
        self.flare_name_list=[]

        self.reflective_flag=False
        self.reflective_dict={}
        self.reflective_list=[]
        self.reflective_name_list=[]


        self.light_flag=False
        self.light_dict={}
        self.light_list=[]
        self.light_name_list=[]

        self.mask_type=mask_type
        self.img_size=transform_base['img_size']

        # --- 硬编码 JSON 文件路径并加载 ---
        # Fixed path for flare coordinates JSON
        fixed_flare_coords_json_path = '/home/user2/wns/deflare/new_dataset.json'
        self.flare_coords = {}
        if osp.exists(fixed_flare_coords_json_path):
            with open(fixed_flare_coords_json_path, 'r', encoding='utf-8') as f:
                self.flare_coords = json.load(f)
            print(f"Loaded flare coordinates from: {fixed_flare_coords_json_path}")
        else:
            print(f"Warning: Flare coordinates JSON not found at {fixed_flare_coords_json_path}. Flares will be randomly placed.")
        # --- 硬编码结束 ---

        # self.transform_base 将只包含翻转操作，RandomCrop 将手动应用。
        # ToTensor 也将在所有 PIL 操作之后手动应用。
        self.transform_base_flips = transforms.Compose([
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip()
                                     ])

        self.flare_transform_params = transform_flare
        # self.transform_flare_initial 不再直接使用，因为 TF.affine 将手动使用

        self.data_ratio=[] 
        print("Base Image Loaded with examples:", len(self.data_list))

    def __getitem__(self, index):
        # load base image
        img_path=self.data_list[index]
        base_img_pil = Image.open(img_path).convert('RGB') # 保持为 PIL Image 格式

        filename = osp.basename(img_path)

        # --- 对 base_img_pil 进行手动 RandomCrop 和坐标映射 ---
        # 获取 base 图像的随机裁剪参数
        i, j, h, w = transforms.RandomCrop.get_params(base_img_pil, output_size=(self.img_size, self.img_size))
        base_img_cropped_pil = TF.crop(base_img_pil, i, j, h, w) # 对 PIL Image 应用裁剪

        # 对 PIL Image 应用随机水平和垂直翻转
        base_img_flipped_pil = self.transform_base_flips(base_img_cropped_pil)

        # 现在转换为 Tensor 并应用 gamma 修正
        to_tensor = transforms.ToTensor()
        base_img = to_tensor(base_img_flipped_pil) # 在这里将 PIL Image 转换为 Tensor
        
        gamma=np.random.uniform(1.8,2.2)
        adjust_gamma=RandomGammaCorrection(gamma)
        adjust_gamma_reverse=RandomGammaCorrection(1/gamma)
        color_jitter=transforms.ColorJitter(brightness=(0.8,3),hue=0.0)

        # 对 Tensor 应用 gamma 修正
        base_img = adjust_gamma(base_img)
        
        sigma_chi=0.01*np.random.chisquare(df=1)
        base_img=Normal(base_img,sigma_chi).sample()
        gain=np.random.uniform(0.5,1.2)
        base_img=gain*base_img
        base_img=torch.clamp(base_img,min=0,max=1)
        
        # 根据 JSON 坐标或随机确定 Flare 位置
        flare_target_x, flare_target_y = None, None
        if filename in self.flare_coords and len(self.flare_coords[filename]) > 0:
            # 从列表中随机选择一个坐标
            orig_cx, orig_cy = random.choice(self.flare_coords[filename])
            
            # 将原始坐标映射到裁剪后的 512x512 空间
            # `j` 是 x 偏移量 (宽度)，`i` 是 y 偏移量 (高度)
            cropped_cx = orig_cx - j 
            cropped_cy = orig_cy - i
            
            flare_target_x = cropped_cx
            flare_target_y = cropped_cy
            
            # 限制坐标确保它们在 512x512 范围内
            flare_target_x = np.clip(flare_target_x, 0, self.img_size - 1)
            flare_target_y = np.clip(flare_target_y, 0, self.img_size - 1)
            
        else:
            # 如果没有特定 Flare 坐标，则在 512x512 输出中随机放置
            flare_target_x = random.randint(0, self.img_size - 1)
            flare_target_y = random.randint(0, self.img_size - 1)
        
        # 加载 Flare 和光源图像
        choice_dataset = random.choices([i for i in range(len(self.flare_list))], self.data_ratio)[0]
        choice_index = random.randint(0, len(self.flare_list[choice_dataset])-1)

        if self.light_flag:
            assert len(self.flare_list)==len(self.light_list), "Error, number of light source and flares dataset no match!"
            for i in range(len(self.flare_list)):
                assert len(self.flare_list[i])==len(self.light_list[i]), f"Error, number of light source and flares no match in {i} dataset!"
            flare_path=self.flare_list[choice_dataset][choice_index]
            light_path=self.light_list[choice_dataset][choice_index]
            light_img_pil=Image.open(light_path).convert('RGB')
            light_img=to_tensor(light_img_pil) # 在这里将 PIL 转换为 Tensor
            light_img=adjust_gamma(light_img)
        else:
            flare_path=self.flare_list[choice_dataset][choice_index]
        flare_img_pil =Image.open(flare_path).convert('RGB')
        
        if self.reflective_flag:
            reflective_path_list=self.reflective_list[choice_dataset]
            if len(reflective_path_list) != 0:
                reflective_path=random.choice(reflective_path_list)
                reflective_img_pil =Image.open(reflective_path).convert('RGB')
            else:
                reflective_img_pil = None

        flare_img=to_tensor(flare_img_pil) # 在这里将 PIL 转换为 Tensor
        flare_img=adjust_gamma(flare_img)
        
        if self.reflective_flag and reflective_img_pil is not None:
            reflective_img=to_tensor(reflective_img_pil) # 在这里将 PIL 转换为 Tensor
            reflective_img=adjust_gamma(reflective_img)
            flare_img = torch.clamp(flare_img+reflective_img,min=0,max=1)

        # 调用修改后的 remove_background 函数，它现在可以正确处理 Tensor
        flare_img=remove_background(flare_img)

        # --- 应用仿射变换到 flare_img，并进行特定平移 ---
        # 获取角度、缩放、剪切的随机参数
        angle = random.uniform(self.flare_transform_params['degrees'][0], self.flare_transform_params['degrees'][1])
        scale_factor = random.uniform(self.flare_transform_params['scale_min'], self.flare_transform_params['scale_max'])
        shear_x = random.uniform(-self.flare_transform_params['shear'], self.flare_transform_params['shear'])
        shear_y = random.uniform(-self.flare_transform_params['shear'], self.flare_transform_params['shear'])
        
        # 计算将 Flare 中心移动到 (flare_target_x, flare_target_y) 的像素平移量
        trans_x = flare_target_x - self.img_size / 2
        trans_y = flare_target_y - self.img_size / 2
        
        if self.light_flag:
            flare_and_light = torch.cat((flare_img, light_img), dim=0) # 6 通道
            
            flare_and_light = TF.affine(
                flare_and_light,
                angle=angle,
                translate=[trans_x, trans_y], # 使用特定的像素平移
                scale=scale_factor,
                shear=[shear_x, shear_y] # TF.affine 期望 shear 是列表/元组
            )
            # 仿射变换后，对 Tensor 应用 CenterCrop 和随机翻转
            flare_and_light = TF.center_crop(flare_and_light, (self.img_size, self.img_size))
            if random.random() > 0.5:
                flare_and_light = TF.hflip(flare_and_light)
            if random.random() > 0.5:
                flare_and_light = TF.vflip(flare_and_light)
            
        else: # 只有 flare_img
            flare_img = TF.affine(
                flare_img,
                angle=angle,
                translate=[trans_x, trans_y], # 使用特定的像素平移
                scale=scale_factor,
                shear=[shear_x, shear_y]
            )
            flare_img = TF.center_crop(flare_img, (self.img_size, self.img_size))
            if random.random() > 0.5:
                flare_img = TF.hflip(flare_img)
            if random.random() > 0.5:
                flare_img = TF.vflip(flare_img)
        # --- 结束 flare_img 的特定平移 ---

        # 改变颜色
        if self.light_flag:
            flare_img, light_img = torch.split(flare_and_light, 3, dim=0)
            # color_jitter 对 Tensor 操作。如果需要，应用于每个分离的图像。
            flare_img = color_jitter(flare_img) # 在这里应用颜色抖动
            light_img = color_jitter(light_img) # 在这里应用颜色抖动
        else:
            flare_img=color_jitter(flare_img)

        # flare 模糊
        blur_transform=transforms.GaussianBlur(21,sigma=(0.1,3.0))
        flare_img=blur_transform(flare_img)
        flare_img=torch.clamp(flare_img,min=0,max=1)
        flare_DC_offset=np.random.uniform(-0.02,0.02)
        flare_img=flare_img+flare_DC_offset
        flare_img=torch.clamp(flare_img,min=0,max=1)

        # 合并图像     
        merge_img=flare_img+base_img
        merge_img=torch.clamp(merge_img,min=0,max=1)
        
        if self.light_flag:
            base_img=base_img+light_img
            base_img=torch.clamp(base_img,min=0,max=1)
            flare_img=flare_img-light_img
            flare_img=torch.clamp(flare_img,min=0,max=1)
        return {'gt': adjust_gamma_reverse(base_img),'flare': adjust_gamma_reverse(flare_img),'lq': adjust_gamma_reverse(merge_img),'gamma':gamma}
        
    def __len__(self):
        return len(self.data_list)
    
    def load_scattering_flare(self,flare_name,flare_path):
        flare_list=[]
        [flare_list.extend(glob.glob(flare_path + '/*.' + e)) for e in self.ext]
        flare_list=sorted(flare_list)
        self.flare_name_list.append(flare_name)
        self.flare_dict[flare_name]=flare_list
        self.flare_list.append(flare_list)
        len_flare_list=len(self.flare_dict[flare_name])
        if len_flare_list == 0:
            print("ERROR: scattering flare images are not loaded properly")
        else:
            print("Scattering Flare Image:",flare_name, " is loaded successfully with examples", str(len_flare_list))
        print("Now we have",len(self.flare_list),'scattering flare images')
    
    def load_light_source(self,light_name,light_path):
        #The number of the light source images should match the number of scattering flares
        light_list=[]
        [light_list.extend(glob.glob(light_path + '/*.' + e)) for e in self.ext]
        light_list=sorted(light_list)
        self.flare_name_list.append(light_name)
        self.light_dict[light_name]=light_list
        self.light_list.append(light_list)
        len_light_list=len(self.light_dict[light_name])

        if len_light_list == 0:
            print("ERROR: Light Source images are not loaded properly")
        else:
            self.light_flag=True
            print("Light Source Image:", light_name, " is loaded successfully with examples", str(len_light_list))
        print("Now we have",len(self.light_list),'light source images')

    def load_reflective_flare(self,reflective_name,reflective_path):
        if reflective_path is None:
            reflective_list=[]
        else:
            reflective_list=[]
            [reflective_list.extend(glob.glob(reflective_path + '/*.' + e)) for e in self.ext]
            reflective_list=sorted(reflective_list)
        self.reflective_name_list.append(reflective_name)
        self.reflective_dict[reflective_name]=reflective_list
        self.reflective_list.append(reflective_list)
        len_reflective_list=len(self.reflective_dict[reflective_name])
        if len_reflective_list == 0:
            print("ERROR: reflective flare images are not loaded properly")
        else:
            self.reflective_flag=True
            print("Reflective Flare Image:",reflective_name, " is loaded successfully with examples", str(len_reflective_list))
        print("Now we have",len(self.reflective_list),'refelctive flare images')

# 请确保 basicsr.utils.registry.DATASET_REGISTRY 在你的环境中是可用的
@DATASET_REGISTRY.register()
class Flare7kpp_Pair_Loader(Flare_Image_Loader):
    def __init__(self, opt):
        super().__init__(opt['image_path'], opt['transform_base'], opt['transform_flare'], opt['mask_type'])
        
        scattering_dict=opt['scattering_dict']
        reflective_dict=opt['reflective_dict']
        light_dict=opt['light_dict']

        if 'data_ratio' not in opt or len(opt['data_ratio'])==0:
            self.data_ratio = [1] * len(scattering_dict)
        else:
            self.data_ratio = opt['data_ratio']

        if len(scattering_dict) !=0:
            for key in scattering_dict.keys():
                self.load_scattering_flare(key,scattering_dict[key])
        if len(reflective_dict) !=0:
            for key in reflective_dict.keys():
                self.load_reflective_flare(key,reflective_dict[key])
        if len(light_dict) !=0:
            for key in light_dict.keys():
                self.load_light_source(key,light_dict[key])

@DATASET_REGISTRY.register()
class Image_Pair_Loader(data.Dataset):
    def __init__(self, opt):
        super(Image_Pair_Loader, self).__init__()
        self.opt = opt
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.paths = glod_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        self.to_tensor=transforms.ToTensor()
        self.gt_size=opt['gt_size']
        self.transform = transforms.Compose([transforms.Resize(self.gt_size), transforms.CenterCrop(self.gt_size), transforms.ToTensor()])

    def __getitem__(self, index):
        gt_path = self.paths['gt'][index]
        lq_path = self.paths['lq'][index]
        img_lq=self.transform(Image.open(lq_path).convert('RGB'))
        img_gt=self.transform(Image.open(gt_path).convert('RGB'))

        return {'lq': img_lq, 'gt': img_gt}

    def __len__(self):
        return len(self.paths['lq'])

@DATASET_REGISTRY.register()
class ImageMask_Pair_Loader(Image_Pair_Loader):
    def __init__(self, opt):
        Image_Pair_Loader.__init__(self,opt)
        self.opt = opt
        self.gt_folder, self.lq_folder,self.mask_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_mask']
        self.paths = glod_from_folder([self.lq_folder, self.gt_folder,self.mask_folder], ['lq', 'gt','mask'])
        self.to_tensor=transforms.ToTensor()
        self.gt_size=opt['gt_size']
        self.transform = transforms.Compose([transforms.Resize(self.gt_size), transforms.CenterCrop(self.gt_size), transforms.ToTensor()])

    def __getitem__(self, index):
        gt_path = self.paths['gt'][index]
        lq_path = self.paths['lq'][index]
        mask_path = self.paths['mask'][index]
        img_lq=self.transform(Image.open(lq_path).convert('RGB'))
        img_gt=self.transform(Image.open(gt_path).convert('RGB'))
        img_mask = self.transform(Image.open(mask_path).convert('RGB'))

        return {'lq': img_lq, 'gt': img_gt,'mask':img_mask}

    def __len__(self):
        return len(self.paths)