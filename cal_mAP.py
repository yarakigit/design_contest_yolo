#===============================================================================
# Training script
#===============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from utils.datasets import _create_data_loader, _create_validation_data_loader
from utils.loss import compute_loss
from utils.utils import plot_graph
from model import load_model
import time
import argparse
import os
import tqdm
import numpy as np
from utils.utils import (non_max_suppression, xywh2xyxy,
                         get_batch_statistics, ap_per_class)

#===============================================================================
# Process arguments
#===============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--weights')
parser.add_argument('--model', default=None)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--decay', type=float, default=0.0005)
parser.add_argument('--data_root', default=os.environ['HOME'] + '/datasets/COCO/2014')
parser.add_argument('--output_model', default='yolo-tiny.pt')
parser.add_argument('--num_classes', type=int, default=80)
parser.add_argument('--trans', action='store_true', default=False)
parser.add_argument('--finetune', action='store_true', default=False)
parser.add_argument('--valid_iou_thres', type=float, default=0.5)
parser.add_argument('--valid_nms_thres', type=float, default=0.5)
parser.add_argument('--valid_conf_thres', type=float, default=0.1)
parser.add_argument('--novalid', action='store_true', default=False)
parser.add_argument('--class_names', default='namefiles/coco.names')
parser.add_argument('--nosave', action='store_true', default=False)
parser.add_argument('--restart', action='store_true', default=False)
parser.add_argument('--dropout', action='store_true', default=False)
args = parser.parse_args()

#===============================================================================
# Define parameters
#===============================================================================
DATA_ROOT    = args.data_root
BATCH_SIZE   = args.batch_size
TEST_PATH   = (DATA_ROOT + '/5k.txt'             
                    if 'COCO' in DATA_ROOT        
                    else DATA_ROOT + '/test.txt')
weights_path = args.weights
NUM_CLASSES  = args.num_classes
IMG_SIZE     = 416
SEP          = True if args.model == "sep" else False
iou_thres    = args.valid_iou_thres
nms_thres    = args.valid_nms_thres
conf_thres   = args.valid_conf_thres
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tensor_type = (torch.cuda.FloatTensor
                    if torch.cuda.is_available()
                    else torch.FloatTensor)

# Load class names from name file
class_file = args.class_names
class_names = []
with open(class_file, 'r') as f:
    class_names = f.read().splitlines()

test_dataloader = _create_data_loader(
    TEST_PATH,
    BATCH_SIZE,
    IMG_SIZE
    )

model = load_model(weights_path, device, NUM_CLASSES, tiny=True, trans=False,
                       finetune=False, use_sep=SEP, dropout=False)

print("Model :", model.__class__.__name__);

print(model)

sample_metrics = []
labels         = []
model.eval()
for _, image, target in tqdm.tqdm(test_dataloader):
    labels += target[:, 1].tolist()
    image = image.type(tensor_type)

    target[:, 2:] = xywh2xyxy(target[:, 2:])
    target[:, 2:] *= IMG_SIZE

    with torch.no_grad():
        outputs = model(image)
        outputs = non_max_suppression(outputs, conf_thres, nms_thres)

    sample_metrics += get_batch_statistics(outputs, target, iou_thres)

TP, pred_scores, pred_labels \
            = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
metrics_output = ap_per_class(TP, pred_scores, pred_labels, labels)

# Calculate AP
precision, recall, AP, f1, ap_class = metrics_output

if NUM_CLASSES != 1:
    ap_table = [['Index', 'Class', 'AP']]
    for i, c in enumerate(ap_class):
        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
    for ap in ap_table:
        print(ap)
mAP = AP.mean() 
print("mAP : %.5f" % mAP)
