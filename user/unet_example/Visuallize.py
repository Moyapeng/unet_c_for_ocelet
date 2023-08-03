import numpy as np
from PIL import Image,ImageDraw
from pylab import *
import cv2
import json
import matplotlib.pyplot as plt
import os
import csv
from user.unet_example.tissue_cell_unet import PytorchUnetCellModel as Model
from util import gcio
# from user.unet_example.unet import PytorchUnetCellModel as Model

str_num=1                            #1~400

json_dir = '/home/f1105/1TB/MYP/ocelot23algo-eval/test/input/metadata.json'
# meta_dataset = gcio.read_json(json_dir)
with open(json_dir, 'r') as f:
    meta_dataset = json.load(f)
model = Model(meta_dataset)

tissue_patch = np.array(Image.open('/home/f1105/1TB/MYP/ocelot23algo-eval/test/input/images/tissue_patches/000.tif'))
cell_patch   = np.array(Image.open('/home/f1105/1TB/MYP/ocelot23algo-eval/test/input/images/cell_patches/000.tif'))

def get_rec_position(json_dir1=json_dir):   #获取矩形框坐标
    js = json.load(open(json_dir1))
    patch_x_offset = js.get('patch_x_offset')
    patch_y_offset = js.get('patch_y_offset')
    a=patch_x_offset=1024*patch_x_offset-128
    b=patch_y_offset=1024*patch_y_offset-128
    return ((a,b),(a+256,b+256))

def get_dot_position(str_num):   #读取细胞标注点
    # csv_dir='ocelot2023/annotations/train/cell'
    # str1=str(str_num).zfill(3)+'.csv'
    # csv_dir=os.path.join(csv_dir,str1)
    cls_to_color_list=[]   #将标签1，2转为颜色字符串

    # csvfile=open(csv_dir)
    # reader = csv.reader(csvfile)
    x_dot_pos=[]
    y_dot_pos=[]
    cls_dot_pos=[]
    # for row in reader:
    #     x_dot_pos.append(int(row[0]))
    #     y_dot_pos.append(int(row[1]))
    #     cls_dot_pos.append(int(row[2]))
    cell_list = model(cell_patch,tissue_patch,0)
    # print(cell_list)
    cell_list = np.array(cell_list,dtype=np.uint)
    # print(cell_list)
    x_dot_pos = np.array(cell_list[:,0],dtype=np.uint)
    y_dot_pos = np.array(cell_list[:,1],dtype=np.uint)
    cls_dot_pos = np.array(cell_list[:,2],dtype=np.uint)
    # print(x_dot_pos,y_dot_pos,cls_to_color_list)
    for i in cls_dot_pos:
        cls_to_color_list.append('red')
    for i in range(len(cls_dot_pos)):
        cls_to_color_list[i]='yellow' if cls_dot_pos[i]==1 else 'cyan'

    return x_dot_pos,y_dot_pos,cls_to_color_list


def get_tumor_position(str_num):   #读取癌变区域坐标
    img_dir = '/home/f1105/1TB/MYP/ocelot23algo-eval/test/input/images/tissue_patches/000.tif'
    # str1 = str(str_num).zfill(3) + '.png'
    # img_dir = os.path.join(img_dir, str1)
    # print(img_dir)
    tissue_anno = Image.open(img_dir)
    tissue_anno = array(tissue_anno)
    # print(tissue_anno)
    x_pos,y_pos = np.where(tissue_anno == 2)
    return list(x_pos),list(y_pos)

def color_map():
    color_map_ = {
        1: [0, 0, 0],  # ignore
        2: [0, 255, 0],  # background
        255: [0, 0, 0],  # building
        }
    return color_map_

def gt2color(str_num):   #将掩码图变为RGB图像
    # read image （np.array）
    img_dir = 'ocelot2023/annotations/train/tissue'
    str1 = str(str_num).zfill(3) + '.png'
    img_dir = os.path.join(img_dir, str1)

    img = cv2.imread(img_dir)
    img_out = np.zeros(img.shape, np.uint8)
    # 获取图像宽*高
    img_x, img_y = img.shape[0], img.shape[1]
    # 得到该数据色彩映射关系
    label_color = color_map()
    # 每行每列像素值依次替换，这里写的简单，你也可以用for ... enumerate...获取索引和值
    for x in range(img_x):
        for y in range(img_y):
            label = img[x][y][0]    # get img label
            img_out[x][y] = label_color[label]
    # cv2.imshow("RGB", img_out)
    return img_out

def merging_mask(origin_img,mask_img,gamma=0.6): #此处掩码图是RGB格式的
    origin_img=np.array(origin_img)
    mask_img = np.array(mask_img)
    img_x,img_y = mask_img.shape[0],mask_img.shape[1]
    merge_img=np.zeros((img_x,img_y,3))
    bg_rgb=[0,0,0]
    for x in range(img_x):
        for y in range(img_y):
            pixel=mask_img[x][y]
            if any(pixel != bg_rgb):
                merge_img[x][y] = gamma * origin_img[x][y] + (1-gamma) * mask_img[x][y]
            else:
                merge_img[x][y] = origin_img[x][y]
            # merge_img[x][y] = gamma * origin_img[x][y] + (1 - gamma) * mask_img[x][y] if pixel!=[0,0,0] else origin_img[x][y]
    # merge_img=merge_img.astype(np.uint8)  #opencv不支持float64
    # merge_img = Image.fromarray(cv2.cvtColor(merge_img,cv2.COLOR_BGR2RGB))
    merge_img = Image.fromarray(merge_img.astype(np.uint8))     #array转PIL

    return merge_img





cell_img  =Image.open('/home/f1105/1TB/MYP/ocelot23algo-eval/test/input/images/cell_patches/000.tif')
tissue_img=Image.open('/home/f1105/1TB/MYP/ocelot23algo-eval/test/input/images/tissue_patches/000.tif')
save_root_path = '/home/f1105/1TB/MYP/ocelot23algo-eval/'
fig_name = str(str_num).zfill(3)+'.jpg'

########################################################

plt.figure()
# plt.subplot(121)  #组织图
# plt.title('tissue')
#
#
# seg_img = gt2color(str_num)   #可视化后的掩码图，掩码图绿色部分为癌变区域
# tissue_img = merging_mask(tissue_img,seg_img,0.6)  #将掩码图覆盖到原图上
#
# # tissue_img=cv2.cvtColor(np.asarray(tissue_img),cv2.COLOR_RGB2BGR)      #PIL转opencv
# # tissue_img=cv2.addWeighted(tissue_img,0.8,seg_img,0.2,gamma=0)
# # tissue_img=Image.fromarray(cv2.cvtColor(tissue_img,cv2.COLOR_BGR2RGB))   #opencv转PIL
#
#
# position=get_rec_position(str_num)            #画框
# tissue_img_draw=ImageDraw.ImageDraw(tissue_img)
# tissue_img_draw.rectangle(position,fill=None,outline='red',width=10)
#
# # tissue_path = os.path.join(save_root_path,'tissue',fig_name)
# # tissue_img.save(tissue_path)
# plt.xticks([])
# plt.yticks([])  #去掉横纵坐标
#
# plt.imshow(tissue_img)


# plt.subplot(122)  #细胞图
plt.title('cell')

cell_img_draw = ImageDraw.ImageDraw(cell_img)

x_dot_pos,ydot_pos,cls_label=get_dot_position(str_num)
# print(get_dot_position(str_num))

plt.scatter(x_dot_pos,ydot_pos,s=40,c=cls_label,marker='.')  #s为点大小


path = os.path.join(save_root_path,fig_name)
# cell_img.save(cell_path)

plt.xticks([])
plt.yticks([])  #去掉横纵坐标
plt.imshow(cell_img)

plt.savefig(path,dpi=1024,bbox_inches='tight')
# plt.close()


plt.show()



######################################################tisuue读取
# tissue_anno=Image.open('ocelot2023/annotations/train/tissue/288.png')
# tissue_anno=array(tissue_anno)
# # tissue_anno[tissue_anno==2]=120
# # tt=np.where(tissue_anno==2)
# # print(list(tt[0]))

##############################################################

# imshow(tissue_anno)
# show()

########################################################读取json文件内容
# js=json.load(open('ocelot2023/metadata.json'))
# x_list=[]
# for i in range(len(js.get('sample_pairs'))):
#     str1=str(i+1).zfill(3)
#     content1=js.get('sample_pairs').get(str1).get('cell').get('x_start')
#     content2 = js.get('sample_pairs').get(str1).get('cell').get('y_start')
#     x_list.append((content1,content2))
#
# print(x_list)
#######################################################################################
