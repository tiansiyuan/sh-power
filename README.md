# 安全帽检测

使用 YOLOv5。

## Create Notebook server on Kubeflow 

- 创建一个 Notebook Server [官方 docker image: kubeflownotebookswg/jupyter-pytorch-cuda-full:v1.6.0] 使用 1 GPU, 4CPUs, 16GB 内存和 10G 卷空间


- [Python](https://www.python.org/), [PyTorch](https://pytorch.org/), 和 [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn)  已经安装

## 下载模型和数据集

- 在 notebook 中 clone 该 repo 并安装 requirements.txt 中的依赖包:

```
!git clone https://github.com/tiansiyuan/sh-power.git
cd sh-pwer/notebook
pip install -r requirements.txt

注意：在运行上述命令后，可能会有类似的警告(Note: you may need to restart the kernel to use updated packages.) You can restart your Jupyter Kernel by simply clicking Kernel > Restart from the Jupyter menu. Note: This will reset your notebook and remove all variables or methods you've defined! Sometimes you'll notice that your notebook is still hanging after you've restart the kernel. If this occurs try refreshing your browser
```

- 我们使用 VOC2007 数据集 (train: *16551 images;*  val: *4952 image*) ，你可以从[这里](https://jhx.japaneast.cloudapp.azure.com/share/VOC2007.zip)获得数据。脚本 `prepare.py` 用于处理数据，将 VOC 标签(label) 格式 (.xml) 转换为 Yolo 标签格式 (.txt) 并切分为训练 (training) 和验证 (validating) 数据。

```
mkdir -p VOCdevkit
cd VOCdevkit

# 下载数据
!wget https://jhx.japaneast.cloudapp.azure.com/share/VOC2007.zip
!unzip VOC2007.zip
Archive:  VOC2007.zip
   creating: VOC2007/
   creating: VOC2007/Annotations/
  inflating: VOC2007/Annotations/000000.xml  
  inflating: VOC2007/Annotations/000002.xml  
  inflating: VOC2007/Annotations/000003.xml  
  inflating: VOC2007/Annotations/000004.xml  
  inflating: VOC2007/Annotations/000005.xml  
  inflating: VOC2007/Annotations/000006.xml  
......
  inflating: VOC2007/JPEGImages/PartB_02404.jpg  

cd ..
!python prepare.py
Probability: 26
Probability: 62
Probability: 78
Probability: 29
Probability: 24
Probability: 65
Probability: 77
Probability: 1
Probability: 17
......
```

- 修改配置文件 <br>
  &ensp; 在 `data` 目录中检查 `hat.yaml` 文件 <br>
  &ensp; 在 `models` 目录中修改 `yolo5s_hat.yaml` 中的 `nc` 参数

```yaml
# hat.yaml
train: ./VOCdevkit/images/train/  # 16551 images
val: ./VOCdevkit/images/val/  # 4952 images

nc: 2  # number of classes
names: ["hat","person"]  # class names

# yolov5s_hat.yaml
nc: 2  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
```

## 模型训练

```
!python train.py     # you can also add '--arguments' to change for your setting

github: skipping check (not a git repository)
YOLOv5 🚀 4e4d2b9 torch 1.8.1+cu111 CUDA:0 (NVIDIA GeForce RTX 3070, 7973.6875MB)

Namespace(adam=False, artifact_alias='latest', batch_size=32, bbox_interval=-1, bucket='', cache_images=False, cfg='models/yolov5s_hat.yaml', data='data/hat.yaml', device='0', entity=None, epochs=50, evolve=False, exist_ok=False, global_rank=-1, hyp='data/hyp.scratch.yaml', image_weights=False, img_size=[640, 640], label_smoothing=0.0, linear_lr=False, local_rank=-1, multi_scale=False, name='exp', noautoanchor=False, nosave=False, notest=False, project='runs/train', quad=False, rect=False, resume=False, save_dir='runs/train/exp5', save_period=-1, single_cls=False, sync_bn=False, total_batch_size=32, upload_dataset=False, weights='yolov5s.pt', workers=8, world_size=1)
tensorboard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
hyperparameters: lr0=0.01, lrf=0.2, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0
wandb: Install Weights & Biases for YOLOv5 logging with 'pip install wandb' (recommended)

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Focus                     [3, 32, 3]                    
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  1    156928  models.common.C3                        [128, 128, 3]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  1    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1    656896  models.common.SPP                       [512, 512, [5, 9, 13]]        
  9                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1     18879  models.yolo.Detect                      [2, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model Summary: 283 layers, 7066239 parameters, 7066239 gradients, 16.5 GFLOPS

Transferred 308/362 items from yolov5s.pt
Scaled weight_decay = 0.0005
Optimizer groups: 62 .bias, 62 conv.weight, 59 other
train: Scanning 'VOCdevkit/labels/train' images and labels... 4549 found, 0 miss/opt/conda/lib/python3.8/site-packages/PIL/TiffImagePlugin.py:845: UserWarning: Corrupt EXIF data.  Expecting to read 4 bytes but only got 0. 
  warnings.warn(str(msg))
train: Scanning 'VOCdevkit/labels/train' images and labels... 6033 found, 0 miss
train: New cache created: VOCdevkit/labels/train.cache
val: Scanning 'VOCdevkit/labels/val' images and labels... 1548 found, 0 missing,
val: New cache created: VOCdevkit/labels/val.cache
Plotting labels... 

autoanchor: Analyzing anchors... anchors/target = 4.24, Best Possible Recall (BPR) = 0.9999
Image sizes 640 train, 640 test
Using 8 dataloader workers
Logging results to runs/train/exp5
Starting training for 50 epochs...

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      0/49     2.91G   0.09378   0.08336   0.01665    0.1938       479       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.468       0.457       0.415       0.169

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      1/49     6.04G   0.06579   0.07102  0.007108    0.1439       390       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.785       0.664       0.738       0.354

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      2/49     6.04G   0.06058   0.06937  0.003524    0.1335       463       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.735        0.72       0.773        0.37

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      3/49     6.04G   0.05835   0.06777  0.002922     0.129       278       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.828       0.763       0.815        0.39

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      4/49     6.04G   0.05321   0.06669  0.002342    0.1222       405       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.867       0.816       0.866       0.474

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      5/49     6.04G   0.04935   0.06548  0.002192     0.117       274       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.906       0.818        0.88       0.499

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      6/49     6.04G   0.04663   0.06439  0.001782    0.1128       572       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.893       0.835       0.891       0.521

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      7/49     6.04G   0.04489   0.06409  0.001657    0.1106       349       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.895       0.822       0.888       0.505

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      8/49     6.04G   0.04453   0.06441  0.001394    0.1103       303       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.898       0.852       0.903       0.537

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
      9/49     6.04G   0.04335   0.06272  0.001248    0.1073       326       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821         0.9       0.846       0.902       0.539

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     10/49     6.04G   0.04282   0.06217  0.001266    0.1063       758       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.913       0.839       0.902       0.542

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     11/49     6.04G   0.04266    0.0621   0.00112    0.1059       258       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821        0.91       0.852       0.907       0.547

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     12/49     6.04G   0.04185   0.06084  0.001211    0.1039       457       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.913       0.849       0.904       0.551

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     13/49     6.04G   0.04183   0.06151  0.001031    0.1044       252       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.916       0.845       0.905       0.558

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     14/49     6.04G    0.0412   0.05972  0.001052     0.102       529       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.925       0.849       0.909       0.562

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     15/49     6.04G   0.04124   0.06067 0.0009574    0.1029       336       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.908       0.858       0.909       0.561

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     16/49     6.04G   0.04065   0.06015 0.0009177    0.1017       517       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.915        0.85       0.912       0.565

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     17/49     6.04G   0.04049   0.05997 0.0008634    0.1013       165       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.899       0.867       0.908       0.559

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     18/49     6.04G   0.04013   0.05961 0.0009647    0.1007       618       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.912       0.865       0.913       0.566

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     19/49     6.04G   0.04006   0.05949 0.0008321    0.1004       316       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.922       0.858       0.916       0.572

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     20/49     6.04G   0.03963   0.05922 0.0008076   0.09967       569       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.908       0.868       0.917       0.572

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     21/49     6.04G   0.03978   0.05891 0.0007754   0.09947       439       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821        0.92        0.85       0.913        0.57

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     22/49     6.04G   0.03931   0.05875 0.0007367   0.09879       261       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.924       0.854       0.912       0.572

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     23/49     6.04G   0.03946   0.05993 0.0006966    0.1001       293       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.917       0.861       0.916       0.579

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     24/49     6.04G   0.03923    0.0589 0.0006993   0.09883       404       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.922       0.859       0.919        0.58

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     25/49     6.04G   0.03908   0.05876 0.0006289   0.09847       321       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.916       0.863       0.917       0.577

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     26/49     6.04G   0.03879   0.05823  0.000679    0.0977       296       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.918       0.866       0.916       0.575

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     27/49     6.04G   0.03885   0.05825 0.0006157   0.09771       464       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821        0.92        0.87       0.919        0.58

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     28/49     6.04G   0.03854   0.05867 0.0006141   0.09782       279       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.917       0.868       0.914       0.578

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     29/49     6.04G   0.03833   0.05818 0.0005556   0.09706       518       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.924       0.862       0.919       0.581

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     30/49     6.04G   0.03845   0.05748 0.0005916   0.09653       527       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.921       0.867        0.92       0.583

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     31/49     6.04G   0.03842   0.05854 0.0005766   0.09754       523       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.925       0.868        0.92       0.583

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     32/49     6.04G   0.03816   0.05767 0.0005051   0.09633       202       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.918        0.87       0.921       0.582

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     33/49     6.04G   0.03799   0.05771 0.0005112   0.09621       401       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.921       0.872        0.92       0.585

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     34/49     6.04G   0.03805   0.05695 0.0004712   0.09547       520       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.929       0.868       0.921       0.585

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     35/49     6.04G   0.03773   0.05711 0.0004824   0.09532       165       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.912       0.872        0.92       0.586

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     36/49     6.04G   0.03767   0.05808 0.0005014   0.09625       498       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.925       0.864       0.922       0.587

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     37/49     6.04G     0.038   0.05869 0.0004919   0.09718       424       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.916       0.872       0.921       0.586

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     38/49     6.04G   0.03739   0.05701 0.0004386   0.09485       227       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.918       0.869       0.922       0.588

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     39/49     6.04G   0.03743    0.0562 0.0004054   0.09403       438       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.936       0.858        0.92       0.587

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     40/49     6.04G   0.03727   0.05635 0.0004607   0.09408       558       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.927       0.867       0.922       0.587

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     41/49     6.04G   0.03725   0.05685 0.0004354   0.09454       383       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.923       0.868       0.921       0.588

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     42/49     6.04G   0.03716   0.05663 0.0003772   0.09416       363       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.917       0.875       0.921       0.587

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     43/49     6.04G   0.03726   0.05685 0.0003887    0.0945       294       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.923       0.872        0.92       0.588

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     44/49     6.04G   0.03715   0.05731 0.0004196   0.09488       188       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.918       0.874       0.921       0.589

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     45/49     6.04G   0.03685   0.05636 0.0003682   0.09357       312       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.923       0.874       0.922       0.589

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     46/49     6.04G    0.0372   0.05651 0.0003705   0.09408       382       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821        0.92       0.877       0.922       0.589

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     47/49     6.04G   0.03709   0.05629 0.0003989   0.09378       447       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.925       0.868        0.92       0.587

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     48/49     6.04G   0.03686   0.05609 0.0003843   0.09334       396       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.923       0.873       0.923        0.59

     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     49/49     6.04G   0.03694   0.05626 0.0004543   0.09366       498       640
               Class      Images      Labels           P           R      mAP@.5
                 all        1548       23821       0.924       0.875       0.925        0.59
                 hat        1548        1785       0.908       0.857       0.915       0.698
              person        1548       22036        0.94       0.894       0.936       0.482
50 epochs completed in 1.950 hours.

Optimizer stripped from runs/train/exp5/weights/last.pt, 14.4MB
Optimizer stripped from runs/train/exp5/weights/best.pt, 14.4MB
```

## 检测与例子

`detect.py` 对几种来源运行推理，使用微调的模型并将结果存入 `runs/detect`。

对于 `VOCdevkit/images` 中的例子图像运行推理：
```
!python detect.py --weight runs/train/exp5/weights/best.pt   --source  VOCdevkit/images/train/000004.jpg

Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device='', exist_ok=False, img_size=640, iou_thres=0.45, name='exp', nosave=False, project='runs/detect', save_conf=False, save_txt=False, source='VOCdevkit/images/train/000004.jpg', update=False, view_img=False, weights=['runs/train/exp5/weights/best.pt'])
YOLOv5 🚀 4e4d2b9 torch 1.8.1+cu111 CUDA:0 (NVIDIA GeForce RTX 3070, 7973.6875MB)

Fusing layers... 
Model Summary: 224 layers, 7056607 parameters, 0 gradients, 16.3 GFLOPS
image 1/1 /home/jovyan/vSphere-machine-learning-extension/examples/end_to_end/helmet_object_detection/notebook/VOCdevkit/images/train/000004.jpg: 448x640 1 hat, 2 persons, Done. (0.018s)
Results saved to runs/detect/exp
Done. (0.031s)

!python detect.py --weight runs/train/exp5/weights/best.pt   --source  VOCdevkit/images/train/000007.jpg --name exp

Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device='', exist_ok=False, img_size=640, iou_thres=0.45, name='exp-1', nosave=False, project='runs/detect', save_conf=False, save_txt=False, source='VOCdevkit/images/train/000007.jpg', update=False, view_img=False, weights=['runs/train/exp5/weights/best.pt'])
YOLOv5 🚀 4e4d2b9 torch 1.8.1+cu111 CUDA:0 (NVIDIA GeForce RTX 3070, 7973.6875MB)

Fusing layers... 
Model Summary: 224 layers, 7056607 parameters, 0 gradients, 16.3 GFLOPS
image 1/1 /home/jovyan/vSphere-machine-learning-extension/examples/end_to_end/helmet_object_detection/notebook/VOCdevkit/images/train/000007.jpg: 448x640 5 hats, Done. (0.017s)
Results saved to runs/detect/exp-1
Done. (0.030s)
```

![Image text](./notebook/imgs/result.jpg)

## 参考资料

- [yolov5](https://github.com/ultralytics/yolov5)
