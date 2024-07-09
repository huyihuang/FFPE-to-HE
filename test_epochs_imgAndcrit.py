import os
from options.test_options import TestOptions

from joblib import Parallel, delayed
import cv2
import pandas as pd
# from skimage.metrics import peak_signal_noise_ratio
# from skimage.metrics import structural_similarity
from sewar.full_ref import psnr, ssim, msssim
from scipy.stats import pearsonr
import shutil
from time import time, ctime
import subprocess

def cal_one_img_crit(imgA_path, imgB_path):
    imgA = cv2.imread(imgA_path, cv2.IMREAD_UNCHANGED)
    imgB = cv2.imread(imgB_path, cv2.IMREAD_UNCHANGED)
    # psnr = peak_signal_noise_ratio(imgA, imgB)
    # ssim = structural_similarity(imgA, imgB, channel_axis=-1) # win_size=7, psi = 1-ssim
    degree_psnr = psnr(imgA, imgB)
    degree_ssim, degree_css = ssim(imgA, imgB) # ws=11
    degree_msssim = msssim(imgA, imgB).real
    # 计算皮尔逊相关系数
    degree_pcc, _ = pearsonr(imgA.flatten(), imgB.flatten())
    return degree_psnr, degree_ssim, degree_css, degree_msssim, degree_pcc

def cal_imgs_crit(imgA_path_list, imgB_path_list, n_jobs=5):
    # 使用 joblib 并行计算
    results = Parallel(n_jobs=n_jobs)(delayed(cal_one_img_crit)(imgA_path, imgB_path) 
                                  for (imgA_path, imgB_path) in zip(imgA_path_list, imgB_path_list))
    psnr_list, ssim_list, css_list, msssim_list, pcc_list = zip(*results)
    return psnr_list, ssim_list, css_list, msssim_list, pcc_list

def save_crit_csv(crit_dict, save_csv_path, round_num=4):
    # 排序,重置索引
    crit_df = pd.DataFrame(crit_dict)
    crit_df = crit_df.sort_values(by=crit_df.keys()[0], ascending=True)
    crit_df = crit_df.reset_index(drop=True)
    # 保留有效小数
    for key in crit_df.columns[1:]:
        crit_df[key] = crit_df[key].round(round_num)
    # 保存csv
    crit_df.to_csv(save_csv_path,index=False)

def cal_epoch_crit(imgs_crit_csv_path):
    epoch_crit = []
    epoch = os.path.basename(imgs_crit_csv_path).split("_")[0]
    epoch_crit.append(int(epoch))

    imgs_crit_csv = pd.read_csv(imgs_crit_csv_path)
    for key in imgs_crit_csv.columns[1:]:
        mean_value = sum(imgs_crit_csv[key])/len(imgs_crit_csv[key])
        epoch_crit.append(mean_value)

    return epoch_crit


opt = TestOptions().parse(save=False)

# 要执行的Python文件
python_file = "test_epochs_img.py"
# 传递的参数字典
arguments = {"--dataroot": opt.dataroot, "--name": opt.name,
             "--phase": opt.phase, "--gpu_ids": str(opt.gpu_ids).strip("[]"),}
# print(str(opt.gpu_ids).strip("[]"))
# input()
# 组装执行命令
gen_img_command = ["python", python_file] + [f"{key}={value}" for key, value in arguments.items()]


# 创建结果文件夹
name_dir = os.path.join(opt.results_dir, opt.name)
criteria_dir = os.path.join(name_dir,'{}_criteria_results'.format(opt.phase))
os.makedirs(criteria_dir,exist_ok=True)

start_epoch = opt.start_epoch
end_epoch = opt.end_epoch
epoch_stride = opt.epoch_stride
n_jobs = opt.nThreads #使用的多线程数

# 逐个计算不同epoch的mean_crit，使用的是cpu
print("start to gen img and cal crit, n_jobs = {}".format(n_jobs))
# 初始化结果
epochs_crit_dict = {'epoch':[],'psnr':[],'ssim':[],'css':[],'msssim':[],'pcc':[]}
epochs_crit_csv_path = os.path.join(name_dir, '{}_epochs_criteria_mean.csv'.format(opt.phase))
for epoch in range(start_epoch, end_epoch+1, epoch_stride):
    # 检查模型是否存在
    model_path = os.path.join(opt.checkpoints_dir,opt.name,'{}_net_G.pth'.format(epoch))
    if not os.path.exists(model_path):
        continue
    # 检查是否算过
    imgs_crit_csv_path = os.path.join(criteria_dir, '{}_criteria.csv'.format(epoch))
    if  os.path.isfile(imgs_crit_csv_path):
        epoch_crit = cal_epoch_crit(imgs_crit_csv_path)
        for i, key in enumerate(epochs_crit_dict.keys()):
            epochs_crit_dict[key].append(epoch_crit[i])
        save_crit_csv(epochs_crit_dict, epochs_crit_csv_path)
        continue

    #检查图片数据是否完整
    web_dir = os.path.join(name_dir, '%s_%s' % (opt.phase, epoch))   
    fakeB_dir  = os.path.join(web_dir, 'images')
    realB_dir = os.path.join(opt.dataroot, opt.phase+'B')
    if os.path.exists(fakeB_dir) and len(os.listdir(fakeB_dir)) == len(os.listdir(realB_dir)):# 数据完整，pass
        pass
    else:# 数据不完整,生成img
        subprocess.run(" ".join(gen_img_command), shell=True )


    # 数据完整，开始计算定量评价指标
    start_time = time()
    print("{} start to epoch {} cal crit".format(ctime(start_time).split(' ')[-2], epoch))

    fakeB_list = sorted(os.listdir(fakeB_dir))
    realB_list = [name.replace('_fake_B', '') for name in fakeB_list]

    fakeB_path_list = [os.path.join(fakeB_dir, name) for name in fakeB_list]
    realB_path_list = [os.path.join(realB_dir, name) for name in realB_list]
    psnr_list, ssim_list, css_list, msssim_list, pcc_list = cal_imgs_crit(fakeB_path_list, realB_path_list, n_jobs)

    # 保存imgs_crit_csv
    imgs_crit_dict = {'img_name':fakeB_list,'psnr':psnr_list, 'ssim':ssim_list,
                      'css':css_list, 'msssim':msssim_list, 'pcc':pcc_list}
    imgs_crit_csv_path = os.path.join(criteria_dir, '{}_criteria.csv'.format(epoch))
    save_crit_csv(imgs_crit_dict, imgs_crit_csv_path)

    # 保存epochs_crit_csv
    epoch_crit = cal_epoch_crit(imgs_crit_csv_path)
    for i, key in enumerate(epochs_crit_dict.keys()):
        epochs_crit_dict[key].append(epoch_crit[i])
    save_crit_csv(epochs_crit_dict, epochs_crit_csv_path)
    
    if epoch % 100 != 0:
        shutil.rmtree(web_dir)

    end_time = time()
    print("epoch {} cal crit time: {}s".format(epoch, end_time-start_time))