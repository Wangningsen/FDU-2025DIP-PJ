import glob
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
import PIL
from basicsr.archs.uformer_arch import Uformer
import argparse
from basicsr.archs.unet_arch import U_Net
from basicsr.utils.flare_util import mkdir,predict_flare_from_6_channel,predict_flare_from_3_channel
import torchvision.transforms as transforms
import os
import gc
import cv2 
from torchvision.transforms import Compose 
import depth_pro 


parser = argparse.ArgumentParser()
parser.add_argument('--input',type=str,default=None)
parser.add_argument('--output',type=str,default=None)
parser.add_argument('--model_type',type=str,default='Uformer')
parser.add_argument('--model_path',type=str,default='checkpoint/flare7kpp/net_g_last.pth')
parser.add_argument('--output_ch',type=int,default=6)
parser.add_argument('--flare7kpp', action='store_const', const=True, default=False) 

args = parser.parse_args()
model_type=args.model_type
images_path=os.path.join(args.input,"*.*")
result_path=args.output
pretrain_dir=args.model_path
output_ch=args.output_ch

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

def load_params(model_path):
    full_model=torch.load(model_path, map_location='cpu') 
    if 'params_ema' in full_model:
        return full_model['params_ema']
    elif 'params' in full_model:
        return full_model['params']
    else:
        return full_model

def demo(images_path,output_path,model_type,output_ch,pretrain_dir,flare7kpp_flag):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    test_path=glob.glob(images_path)
    result_path=output_path 
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    if model_type=='Uformer':
        model=Uformer(img_size=512,img_ch=4,output_ch=output_ch).to(device)
    elif model_type=='U_Net' or model_type=='U-Net':
        model=U_Net(img_ch=4,output_ch=output_ch).to(device) 
    else:
        assert False, "This model is not supported!!"

    model.load_state_dict(load_params(pretrain_dir))
    model.eval() 

    depth_model, depth_transform = depth_pro.create_model_and_transforms()
    depth_model.eval()
    depth_model.to(device)
    
    # 注意：使用 infer 方法时，我们不再直接使用 depth_model.img_size 来手动插值输入
    # 因为 infer 方法内部会处理尺寸调整。
    print(f"Loaded de-flaring model: {model_type} from {pretrain_dir}")
    print(f"Loaded depth model: {depth_pro.__name__}")
    # print(f"Depth model expects input size: {depth_model.img_size}x{depth_model.img_size}") # 可以注释掉或用于调试

    mkdir(os.path.join(result_path, "depth"))
    mkdir(os.path.join(result_path, "flare"))
    mkdir(os.path.join(result_path, "input")) 
    mkdir(os.path.join(result_path, "blend"))
    if not flare7kpp_flag:
        mkdir(os.path.join(result_path, "deflare")) 

    print(f"Processing {len(test_path)} images...")

    for i,image_path in tqdm(enumerate(test_path)):
        image_name_with_ext = os.path.basename(image_path) 
        image_name, _ = os.path.splitext(image_name_with_ext) 
        # image_name = image_name[:6] 


        depth_image_path = os.path.join(result_path, "depth", image_name + "_depth.png")
        flare_path = os.path.join(result_path, "flare", image_name + "_flare.png")
        merge_path = os.path.join(result_path, "input", image_name + "_input.png") 
        blend_path = os.path.join(result_path, "blend", image_name + "_blend.png")
        if not flare7kpp_flag:
            deflare_path = os.path.join(result_path, "deflare", str(i).zfill(5) + "_deflare.png") 

        img_pil = Image.open(image_path).convert('RGB')
        
        original_img_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to(device)

        # **START OF MODIFIED SECTION FOR DEPTH ESTIMATION**
        # 1. 使用 depth_transform 预处理原始 PIL Image
        #    depth_transform 已经包含 ToTensor() 和 Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        #    它的输出是 (C, H, W) 的张量
        input_for_depth_transform = depth_transform(img_pil) 
        # input_for_depth_transform 现在是 (C, H, W)
        
        # 2. 为 depth_model.infer() 准备输入
        #    infer 方法期望 (N, C, H, W)
        #    infer 方法内部会处理尺寸调整，所以不需要手动插值
        sample_for_depth_infer = input_for_depth_transform.unsqueeze(0).to(device)

        with torch.no_grad():
            # 使用 infer 方法获取米制深度和焦距
            output_from_depth_pro = depth_model.infer(sample_for_depth_infer)
            
            # 提取深度图 (米制)
            # infer 方法返回的 depth 是 (H, W) 的张量，需要 Unsqueeze 增加维度
            depth_meter_tensor = output_from_depth_pro["depth"] # 这是一个 (H, W) 的张量

            # 将深度图归一化到 0-1 范围，以便可视化 (通常深度值可能很大)
            # 这里需要根据深度范围进行适当的归一化，例如，如果深度范围是 0-100 米，
            # 可以将 0-100 映射到 0-255。
            # 或者，如果希望像之前那样裁剪到 3000m，就使用 prediction / 3000.0
            
            # 由于 infer 返回的是米制深度，我们直接使用它，并进行适当的归一化保存
            # 这里的归一化方式与您之前的类似，但现在是处理米制深度
            depth_to_process = depth_meter_tensor.cpu().numpy() # 转换为 NumPy 数组

            # 裁剪并归一化以用于可视化
            depth_to_process = np.clip(depth_to_process, 0, 300) # 将深度值裁剪到 0-3000 米
            out_normalized_for_vis = depth_to_process / 300.0 # 归一化到 0-1 范围

            depth_to_save = (out_normalized_for_vis * 255).astype(np.uint8)
            PIL.Image.fromarray(depth_to_save).save(depth_image_path)
            
            # 用于下游 Uformer 模型的深度通道，依然是 0-1 范围，且需要 (N, 1, H, W)
            # 注意：这里的 depth_channel_tensor 需要与 original_img_tensor 的尺寸匹配
            # infer 方法返回的深度图尺寸与原始图像尺寸一致，所以这里可以直接用
            depth_channel_tensor = torch.from_numpy(out_normalized_for_vis).unsqueeze(0).unsqueeze(0).to(device).float()
            # **END OF MODIFIED SECTION FOR DEPTH ESTIMATION**
            
        merge_img = torch.cat((original_img_tensor, depth_channel_tensor), dim=1) 

        with torch.no_grad():
            output_img=model(merge_img) 
            
            gamma=torch.Tensor([2.2]).to(device) 

            if output_ch==6:
                deflare_img,flare_img_predicted,merge_img_predicted=predict_flare_from_6_channel(output_img,gamma)
            elif output_ch==3:
                flare_mask=torch.zeros_like(original_img_tensor) 
                deflare_img,flare_img_predicted=predict_flare_from_3_channel(output_img,flare_mask,output_img,original_img_tensor,original_img_tensor,gamma)
            else:
                assert False, "This output_ch is not supported!!"
            
            torchvision.utils.save_image(original_img_tensor, merge_path) 
            torchvision.utils.save_image(flare_img_predicted, flare_path) 
            
            if flare7kpp_flag:
                torchvision.utils.save_image(deflare_img, blend_path) 
            else:
                torchvision.utils.save_image(deflare_img, deflare_path)
        
        del img_pil, original_img_tensor, input_for_depth_transform, sample_for_depth_infer, \
            output_from_depth_pro, depth_meter_tensor, depth_to_process, out_normalized_for_vis, \
            depth_channel_tensor, merge_img, output_img
        if 'deflare_img' in locals(): del deflare_img
        if 'flare_img_predicted' in locals(): del flare_img_predicted
        if 'merge_img_predicted' in locals(): del merge_img_predicted 
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        
if __name__ == "__main__":
    demo(images_path,result_path,model_type,output_ch,pretrain_dir,args.flare7kpp)