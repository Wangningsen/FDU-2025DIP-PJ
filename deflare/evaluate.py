# import argparse
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from skimage.metrics import structural_similarity as compare_ssim
# from skimage.metrics import mean_squared_error as compare_mse
# from skimage import io
# from torchvision.transforms import ToTensor
# import numpy as np
# from glob import glob
# import lpips
# from tqdm import tqdm

# import warnings
# warnings.filterwarnings("ignore")

# def compare_lpips(img1, img2, loss_fn_alex):
#     to_tensor=ToTensor()
#     img1_tensor = to_tensor(img1).unsqueeze(0)
#     img2_tensor = to_tensor(img2).unsqueeze(0)
#     output_lpips = loss_fn_alex(img1_tensor.cuda(), img2_tensor.cuda())
#     return output_lpips.cpu().detach().numpy()[0,0,0,0]

# def compare_score(img1,img2,img_seg):
#     # Return the G-PSNR, S-PSNR, Global-PSNR and Score
#     # This module is for the MIPI 2023 Challange: https://codalab.lisn.upsaclay.fr/competitions/9402
#     mask_type_list=['glare','streak','global']
#     metric_dict={'glare':0,'streak':0,'global':0}
#     for mask_type in mask_type_list:
#         mask_area,img_mask=extract_mask(img_seg)[mask_type]
#         if mask_area>0:
#             img_gt_masked=img1*img_mask
#             img_input_masked=img2*img_mask
#             input_mse=compare_mse(img_gt_masked, img_input_masked)/(255*255*mask_area)
#             input_psnr=10 * np.log10((1.0 ** 2) / input_mse)
#             metric_dict[mask_type]=input_psnr
#         else:
#             metric_dict.pop(mask_type)
#     return metric_dict

# def extract_mask(img_seg):
#     # Return a dict with 3 masks including streak,glare,global(whole image w/o light source), masks are returned in 3ch. 
#     # glare: [255,255,0]
#     # streak: [255,0,0]
#     # light source: [0,0,255]
#     # others: [0,0,0]
#     mask_dict={}
#     streak_mask=(img_seg[:,:,0]-img_seg[:,:,1])/255
#     glare_mask=(img_seg[:,:,1])/255
#     global_mask=(255-img_seg[:,:,2])/255
#     mask_dict['glare']=[np.sum(glare_mask)/(512*512),np.expand_dims(glare_mask,2).repeat(3,axis=2)] #area, mask
#     mask_dict['streak']=[np.sum(streak_mask)/(512*512),np.expand_dims(streak_mask,2).repeat(3,axis=2)] 
#     mask_dict['global']=[np.sum(global_mask)/(512*512),np.expand_dims(global_mask,2).repeat(3,axis=2)] 
#     return mask_dict

# def calculate_metrics(args):
#     loss_fn_alex = lpips.LPIPS(net='alex').cuda()
#     gt_folder = args['gt'] + '/*'
#     input_folder = args['input'] + '/*'
#     gt_list = sorted(glob(gt_folder))
#     input_list = sorted(glob(input_folder))
#     if args['mask'] is not None:
#         mask_folder = args['mask'] + '/*'
#         mask_list= sorted(glob(mask_folder))

#     assert len(gt_list) == len(input_list)
#     n = len(gt_list)

#     ssim, psnr, lpips_val = 0, 0, 0
#     score_dict={'glare':0,'streak':0,'global':0,'glare_num':0,'streak_num':0,'global_num':0}
#     for i in tqdm(range(n)):
#         img_gt = io.imread(gt_list[i])
#         img_input = io.imread(input_list[i])
#         ssim += compare_ssim(img_gt, img_input, multichannel=True)
#         psnr += compare_psnr(img_gt, img_input, data_range=255)
#         lpips_val += compare_lpips(img_gt, img_input, loss_fn_alex)
#         if args['mask'] is not None:
#             img_seg=io.imread(mask_list[i])
#             metric_dict=compare_score(img_gt,img_input,img_seg)
#             for key in metric_dict.keys():
#                 score_dict[key]+=metric_dict[key]
#                 score_dict[key+'_num']+=1
#     ssim /= n
#     psnr /= n
#     lpips_val /= n
#     print(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips_val}")
#     if args['mask'] is not None:
#         for key in ['glare','streak','global']:
#             if score_dict[key+'_num'] == 0:
#                 assert False, "Error, No mask in this type!"
#             score_dict[key]/= score_dict[key+'_num']
#         score_dict['score']=1/3*(score_dict['glare']+score_dict['global']+score_dict['streak'])
#         print(f"Score: {score_dict['score']}, G-PSNR: {score_dict['glare']}, S-PSNR: {score_dict['streak']}, Global-PSNR: {score_dict['global']}")
        

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input',type=str,default=None)
#     parser.add_argument('--gt',type=str,default=None)
#     parser.add_argument('--mask',type=str,default=None)
#     args = vars(parser.parse_args())
#     calculate_metrics(args)

import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage import io
from torchvision.transforms import ToTensor
import numpy as np
from glob import glob
import lpips
from tqdm import tqdm
import os # 确保导入 os 模块

import warnings
warnings.filterwarnings("ignore")

def compare_lpips(img1, img2, loss_fn_alex):
    to_tensor = ToTensor()
    img1_tensor = to_tensor(img1).unsqueeze(0)
    img2_tensor = to_tensor(img2).unsqueeze(0)

    # --------------------------- 修正点 ---------------------------
    # 将张量明确地移动到 CUDA 设备，因为 loss_fn_alex 模型已经在 CUDA 设备上
    img1_tensor = img1_tensor.cuda() 
    img2_tensor = img2_tensor.cuda()
    # ------------------------------------------------------------

    output_lpips = loss_fn_alex(img1_tensor, img2_tensor)
    return output_lpips.cpu().detach().numpy()[0,0,0,0]
def compare_score(img1,img2,img_seg):
    # 返回 G-PSNR, S-PSNR, Global-PSNR 和 Score
    # 此模块用于 MIPI 2023 挑战赛: https://codalab.lisn.upsaclay.fr/competitions/9402
    mask_type_list=['glare','streak','global']
    metric_dict={'glare':0,'streak':0,'global':0}
    for mask_type in mask_type_list:
        mask_area,img_mask=extract_mask(img_seg)[mask_type]
        if mask_area > 0:
            img_gt_masked = img1 * img_mask
            img_input_masked = img2 * img_mask
            # 修正 MSE 计算，确保分母正确
            input_mse = compare_mse(img_gt_masked, img_input_masked) / mask_area
            # PSNR 公式使用 255 作为数据范围
            input_psnr = 10 * np.log10((255.0 ** 2) / input_mse)
            metric_dict[mask_type] = input_psnr
        else:
            metric_dict.pop(mask_type)
    return metric_dict

def extract_mask(img_seg):
    # 返回一个包含 streak、glare、global（不含光源的整个图像）的三个掩码的字典，掩码以 3 通道形式返回。
    # glare: [255,255,0]
    # streak: [255,0,0]
    # light source: [0,0,255]
    # others: [0,0,0]
    mask_dict={}
    streak_mask=(img_seg[:,:,0]-img_seg[:,:,1])/255
    glare_mask=(img_seg[:,:,1])/255
    global_mask=(255-img_seg[:,:,2])/255
    mask_dict['glare']=[np.sum(glare_mask)/(512*512),np.expand_dims(glare_mask,2).repeat(3,axis=2)] # 区域, 掩码
    mask_dict['streak']=[np.sum(streak_mask)/(512*512),np.expand_dims(streak_mask,2).repeat(3,axis=2)]
    mask_dict['global']=[np.sum(global_mask)/(512*512),np.expand_dims(global_mask,2).repeat(3,axis=2)]
    return mask_dict

def calculate_metrics(args):
    # --------------------------- 修正点 ---------------------------
    # 初始化 LPIPS 模型，并确保它在 GPU 上
    loss_fn_alex = lpips.LPIPS(net='alex').cuda() 
    # ------------------------------------------------------------

    gt_folder = args['gt'] + '/*'
    input_folder = args['input'] + '/*'
    gt_list = sorted(glob(gt_folder))
    input_list = sorted(glob(input_folder))
    if args['mask'] is not None:
        mask_folder = args['mask'] + '/*'
        mask_list= sorted(glob(mask_folder))

    assert len(gt_list) == len(input_list)
    n = len(gt_list)

    ssim_total, psnr_total, lpips_val_total = 0, 0, 0
    processed_image_count = 0

    score_dict={'glare':0,'streak':0,'global':0,'glare_num':0,'streak_num':0,'global_num':0}
    for i in tqdm(range(n)):
        gt_path = gt_list[i]
        input_path = input_list[i]

        gt_image_name = os.path.basename(gt_path)
        input_image_name = os.path.basename(input_path)

        try:
            img_gt = io.imread(gt_path)
            img_input = io.imread(input_path)

            print(f"\n--- DEBUG: 正在处理图片 {i+1}/{n} ---")
            print(f"DEBUG: GT 图片 {gt_image_name} 加载后形状: {img_gt.shape}, 维度: {img_gt.ndim}")
            print(f"DEBUG: Input 图片 {input_image_name} 加载后形状: {img_input.shape}, 维度: {img_input.ndim}")

        except Exception as e:
            print(f"错误: 加载图片失败: {gt_path} 或 {input_path}。错误信息: {e}")
            continue

        if img_gt.ndim < 2 or img_input.ndim < 2 or \
           img_gt.shape[0] < 7 or img_gt.shape[1] < 7 or \
           img_input.shape[0] < 7 or img_input.shape[1] < 7:
            print(f"警告: 图片 {gt_image_name} (形状: {img_gt.shape}) 或 {input_image_name} (形状: {img_input.shape}) 尺寸过小或维度不足，无法计算 SSIM。跳过当前图片对的指标计算。")
            continue

        try:
            # SSIM 参数已在上次修改中添加，这里无需再次修改
            ssim_val = compare_ssim(img_gt, img_input, multichannel=True, channel_axis=-1, data_range=255)
            psnr_val = compare_psnr(img_gt, img_input, data_range=255)
            lpips_val = compare_lpips(img_gt, img_input, loss_fn_alex) # 调用修正后的 compare_lpips

            ssim_total += ssim_val
            psnr_total += psnr_val
            lpips_val_total += lpips_val
            processed_image_count += 1

            if args['mask'] is not None:
                mask_path = mask_list[i]
                try:
                    img_seg = io.imread(mask_path)
                    if img_seg.ndim < 2 or img_seg.shape[0] < 7 or img_seg.shape[1] < 7:
                        print(f"警告: 掩码图片 {os.path.basename(mask_path)} (形状: {img_seg.shape}) 尺寸过小或维度不足。跳过其得分计算。")
                        continue
                    metric_dict = compare_score(img_gt, img_input, img_seg)
                    for key in metric_dict.keys():
                        score_dict[key] += metric_dict[key]
                        score_dict[key+'_num'] += 1
                except Exception as e:
                    print(f"错误: 处理图片 {gt_image_name} 的掩码 ({mask_path}) 失败。错误信息: {e}")
                    pass

        except Exception as e:
            print(f"错误: 计算图片对指标失败: GT: {gt_image_name}, Input: {input_image_name}。错误信息: {e}")
            continue

    if processed_image_count > 0:
        ssim_avg = ssim_total / processed_image_count
        psnr_avg = psnr_total / processed_image_count
        lpips_avg = lpips_val_total / processed_image_count
        print(f"平均 PSNR: {psnr_avg}, 平均 SSIM: {ssim_avg}, 平均 LPIPS: {lpips_avg}")
    else:
        print("没有成功处理任何图片来计算指标。请检查图片路径和文件。")

    if args['mask'] is not None:
        for key in ['glare','streak','global']:
            if score_dict[key+'_num'] == 0:
                print(f"警告: 没有找到或处理任何 '{key}' 类型的掩码。")
                score_dict[key] = np.nan
            else:
                score_dict[key] /= score_dict[key+'_num']

        if all(key in score_dict and not np.isnan(score_dict[key]) for key in ['glare', 'global', 'streak']):
            score_dict['score'] = 1/3 * (score_dict['glare'] + score_dict['global'] + score_dict['streak'])
            print(f"得分: {score_dict['score']}, G-PSNR: {score_dict['glare']}, S-PSNR: {score_dict['streak']}, Global-PSNR: {score_dict['global']}")
        else:
            print("由于缺少掩码类型或处理错误，无法计算总分。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',type=str,default=None)
    parser.add_argument('--gt',type=str,default=None)
    parser.add_argument('--mask',type=str,default=None)
    args = vars(parser.parse_args())
    calculate_metrics(args)