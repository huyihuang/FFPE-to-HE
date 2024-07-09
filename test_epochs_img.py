import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from time import time, ctime
from PIL import Image

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

test_dir = os.path.join(opt.dataroot, opt.phase+'A')
test_list = os.listdir(test_dir)
opt.loadSize, _ = Image.open(os.path.join(test_dir,test_list[0])).size
opt.how_many = len(test_list)
opt.no_instance = True

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
visualizer = Visualizer(opt)

# 创建结果文件夹
name_dir = os.path.join(opt.results_dir, opt.name)
criteria_dir = os.path.join(name_dir,'{}_criteria_results'.format(opt.phase))
os.makedirs(criteria_dir,exist_ok=True)

start_epoch = opt.start_epoch
end_epoch = opt.end_epoch
epoch_stride = opt.epoch_stride

# 生成所有的epoch的imgs，之后就可以空出gpu
print("start to save imgs")
for epoch in range(start_epoch, end_epoch+1, epoch_stride):
    # 检查模型是否存在
    model_path = os.path.join(opt.checkpoints_dir,opt.name,'{}_net_G.pth'.format(epoch))
    if not os.path.exists(model_path):
        continue
    opt.which_epoch = epoch
    # 检查是否算过
    imgs_crit_csv_path = os.path.join(criteria_dir, '{}_criteria.csv'.format(opt.which_epoch))
    if  os.path.isfile(imgs_crit_csv_path):
        continue
    #检查图片数据是否完整
    web_dir = os.path.join(name_dir, '%s_%s' % (opt.phase, opt.which_epoch))
    fakeB_dir  = os.path.join(web_dir, 'images')
    realB_dir = os.path.join(opt.dataroot, opt.phase+'B')
    if os.path.exists(fakeB_dir) and len(os.listdir(fakeB_dir)) == len(os.listdir(realB_dir)): # 数据完整，不重复生成图像
        continue

    # 生成虚拟图像
    start_time = time()
    print("{} start to save epoch {} imgs".format(ctime(start_time).split(' ')[-2], opt.which_epoch))
    # create website
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    if not opt.engine and not opt.onnx:
        model = create_model(opt)
        if opt.data_type == 16:
            model.half()
        elif opt.data_type == 8:
            model.type(torch.uint8)
                
        if opt.verbose:
            print(model)
    else:
        from run_engine import run_trt_engine, run_onnx

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        if opt.data_type == 16:
            data['label'] = data['label'].half()
            data['inst']  = data['inst'].half()
        elif opt.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst']  = data['inst'].uint8()
        if opt.export_onnx:
            print ("Exporting to ONNX: ", opt.export_onnx)
            assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
            torch.onnx.export(model, [data['label'], data['inst']],
                            opt.export_onnx, verbose=True)
            exit(0)
        minibatch = 1 
        if opt.engine:
            generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
        elif opt.onnx:
            generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
        else:        
            generated = model.inference(data['label'], data['inst'], data['image'])
            
        visuals = OrderedDict([('fake_B', util.tensor2im(generated.data[0])),])
        img_path = data['path']
        # print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)
    webpage.save()
    end_time = time()
    print("epoch {} Img save time: {}s".format(opt.which_epoch, end_time-start_time))
