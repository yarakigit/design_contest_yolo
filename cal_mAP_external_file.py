#===============================================================================
# Inference on an image
#===============================================================================

from utils.transforms_detect import resize_aspect_ratio
import torch
import os
import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from torchvision import transforms
import cv2
from model import load_model
from utils.utils import non_max_suppression, plot_distrib
import time
from tqdm import tqdm
import glob
#matplotlib.use('TkAgg')

# Extract output of hidden layer
def extract(target, inputs):
    feature = None

    def forward_hook(module, inputs, outputs):
        global features
        features = outputs.detach().clone()

    handle = target.register_forward_hook(forward_hook)

    model.eval()
    model(inputs, distri_array=activate_distrib, debug=args.debug)

    handle.remove()

    return features

#===============================================================================
# Process arguments
#===============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model', default=None)
parser.add_argument('--weights', default='weights/yolov3-tiny.pt')
parser.add_argument('--map_image_dir')
parser.add_argument('--map_image_dir_out')
parser.add_argument('--file_type')
parser.add_argument('--conf_thres', type=float, default=0.1)
parser.add_argument('--nms_thres', type=float, default=0.4)
parser.add_argument('--nms_vis_en', action='store_true', default=False)
parser.add_argument('--output_image', default='output.jpg')
parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--class_names', default='namefiles/car.names')
parser.add_argument('--quant', action='store_true', default=False)
parser.add_argument('--nogpu', action='store_true', default=False)
parser.add_argument('--notiny', action='store_true', default=False)
parser.add_argument('--distrib', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--merge', action='store_true', default=False)
args = parser.parse_args()

#===============================================================================
# Define parameters
#===============================================================================
weights_path       = args.weights
DIR                = args.map_image_dir
OUT_DIR            = args.map_image_dir_out
FILE_TYPE          = args.file_type
conf_thres   = args.conf_thres
nms_thres    = args.nms_thres
nms_vis_en   = args.nms_vis_en
NUM_CLASSES  = args.num_classes
name_file    = args.class_names
NO_GPU       = args.nogpu
EN_TINY      = not args.notiny
SEP          = True if args.model == "sep" else False
MERGE        = args.merge

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if NO_GPU or args.quant:
    device = torch.device("cpu")
tensor_type = (torch.cuda.FloatTensor
                if (torch.cuda.is_available() and not NO_GPU)
                else torch.FloatTensor)
if args.quant:
    tensor_type = torch.ByteTensor

# Load class names from name file
class_names = []
with open(name_file, 'r') as f:
    class_names = f.read().splitlines()

# Create model
model = load_model(weights_path, device, merge=MERGE, tiny=EN_TINY,
                   num_classes=NUM_CLASSES, quant=args.quant, jit=True,
                   use_sep=SEP)

#input map image dir
total_num = sum(os.path.isfile(os.path.join(DIR+'/images', name)) for name in os.listdir(DIR+'/images'))
bar = tqdm(total=total_num)
print(total_num)
bar.set_description('rate')

for pathAndFilename in glob.iglob(os.path.join(DIR+'/images', "*."+FILE_TYPE)):
        start = time.time();
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        ## detect process
        # Load image
        image_path = DIR+'/images/'+title+ext
        input_image = cv2.imread(image_path)                                                     
        rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)                                 
        image = transforms.ToTensor()(input_image)      # transform to [0-1]                     
                                                                                                 
        image_load_t = time.time()                                                               
                                                                                                 
        image = resize_aspect_ratio(image)                                                       
        image = torch.from_numpy(image)                                                          
        image = image.to(device)                                                                 
        image = image.permute(2, 0, 1)                                                           
        image = image[[2,1,0],:,:]                                                               
        image = image.unsqueeze(0)                                                               
                                                                                                 
        image_convert_t = time.time()                                                            
                                                                                                 
        model.eval()                                                                             
        if args.distrib:                                                                         
            activate_distrib = np.zeros(10)                                                      
                                                                                                 
        # Forward                                                                                
        output = model(image)                                                                    
                                                                                                 
        # l1_conv_dw_output = extract(model.conv1.conv_dw, image)                                
                                                                                                 
        if args.distrib:                                                                         
            print("all :", activate_distrib[0])                                                  
            print("distrib :", "%.2f" % ((activate_distrib[1] / activate_distrib[0]) * 100), "%")
            print("distrib :", "%.2f" % ((activate_distrib[2] / activate_distrib[0]) * 100), "%")
            print("distrib :", "%.2f" % ((activate_distrib[3] / activate_distrib[0]) * 100), "%")
            print("distrib :", "%.2f" % ((activate_distrib[4] / activate_distrib[0]) * 100), "%")
            print("distrib :", "%.2f" % ((activate_distrib[5] / activate_distrib[0]) * 100), "%")
            print("distrib :", "%.2f" % ((activate_distrib[6] / activate_distrib[0]) * 100), "%")
            print("distrib :", "%.2f" % ((activate_distrib[7] / activate_distrib[0]) * 100), "%")
            print("distrib :", "%.2f" % ((activate_distrib[8] / activate_distrib[0]) * 100), "%")
            print("distrib :", "%.2f" % ((activate_distrib[8] / activate_distrib[0]) * 100), "%")
                                                                                                 
            plot_distrib(activate_distrib)                                                       
                                                                                                 
        inference_t = time.time()                                                                
                                                                                                 
        # Apply NMS to inference output                                                          
        # This output is already scaled to 0~416                                                 
        output = non_max_suppression(output, conf_thres, nms_thres)                              
                                                                                                 
        nms_t = time.time()                                                                      
                                                                                                 
        output = output[0]                                                                       
        ##strdout###print("output :", output.shape)                                                          
                                                                                                 
        # Scale box coordinates according to the original size                                   
        orig_h, orig_w = input_image.shape[0:2]                                                  
        pad_x = max(orig_h - orig_w, 0) * (416 / max(orig_h, orig_w))                            
        pad_y = max(orig_w - orig_h, 0) * (416 / max(orig_h, orig_w))                            
                                                                                                 
        unpad_h = 416 - pad_y                                                                    
        unpad_w = 416 - pad_x                                                                    
                                                                                                 
        # Restore the original size in consideration of padding                                  
        output[:, 0] = ((output[:, 0] - pad_x // 2) / unpad_w) * orig_w                          
        output[:, 1] = ((output[:, 1] - pad_y // 2) / unpad_h) * orig_h                          
        output[:, 2] = ((output[:, 2] - pad_x // 2) / unpad_w) * orig_w                          
        output[:, 3] = ((output[:, 3] - pad_y // 2) / unpad_h) * orig_h                          
                                                                                                 
        # Base image to draw                                                                     
        plt.figure()                                                                             
        fig, ax = plt.subplots(1)                                                                
        ax.imshow(rgb_image)                                                                     
                                                                                                 
        # Create color map                                                                       
        cmap = plt.get_cmap('tab20b')       # 'tab20b' is one of colormap names                  
        colors = [cmap(i) for i in np.linspace(0, 1, NUM_CLASSES)]                               
                                                                                                 
        # (Arbitrary)                                                                            
        bbox_colors = random.sample(colors, NUM_CLASSES)                                         
                                                                                                 
        #===============================================================================         
        # Create rectangles and labels using inference results                                   
        #===============================================================================         
        for x_min, y_min, x_max, y_max, conf, class_pred in output:                              
            box_w = x_max - x_min                                                                
            box_h = y_max - y_min                                                                
                                                                                                 
            color = bbox_colors[int(class_pred)]                                                 
            bbox = patches.Rectangle((x_min, y_min), box_w, box_h, linewidth=2,                  
                                edgecolor=color, facecolor='None')                               
            ax.add_patch(bbox)                                                                   
                                                                                                 
            # Label                                                                              
            plt.text(x_min, y_min, s=class_names[int(class_pred)], color='white',                
                                verticalalignment='top', bbox={'color': color, 'pad':0})         
                                                                                                 
        end = time.time()                                                                        
        ##stdout###print("elapsed time = %.4f sec" % (end - start))                                         
        ##stdout###print("items :")                                                                         
        ##stdout###print(" image_load : %.4f sec" % (image_load_t - start))                                 
        ##stdout###print(" image_convert : %.4f sec" % (image_convert_t - image_load_t))                    
        ##stdout###print(" inference : %.4f sec" % (inference_t - image_convert_t))                         
        ##stdout###print(" nms : %.4f sec" % (nms_t - inference_t))                                         
        ##stdout###print(" plot : %.4f sec" % (end - nms_t))                                                
                                                                                                 
        #===============================================================================         
        # Draw                                                                                   
        #===============================================================================         
        plt.axis("off")
        output_path = OUT_DIR+'/images/'+title+ext
        if nms_vis_en:
            plt.savefig(output_path,format=FILE_TYPE)                                                                 
        #plt.show()
        plt.clf()
        plt.close()                                                                              

        ### gen text-flie for mAP
        output_txt_for_mAP = ""
        output_path_label = OUT_DIR+'/labels/'+title+".txt"
        if output.shape[0]==0:## no bbox
            w_file = open(output_path_label, mode="w")
            w_file.write(output_txt_for_mAP)
            w_file.close()
        for x_min, y_min, x_max, y_max, conf, class_pred in output:
            output_txt_for_mAP += "{cls:d} {conf:f} {x1:d} {y1:d} {x2:d} {y2:d}\n".format(cls=int(class_pred),conf=float(conf),x1=int(x_min),y1=int(y_min),x2=int(x_max),y2=int(y_max))
            w_file = open(output_path_label, mode="w")
            w_file.write(output_txt_for_mAP)
            w_file.close()
        bar.update(1)
bar.close
