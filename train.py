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
EPOCHS       = args.epochs
LR           = args.lr
TRAIN_PATH   = (DATA_ROOT + '/trainvalno5k.txt' 
                    if 'COCO' in DATA_ROOT
                    else DATA_ROOT + '/train.txt')
VALID_PATH   = (DATA_ROOT + '/5k.txt' 
                    if 'COCO' in DATA_ROOT
                    else DATA_ROOT + '/valid.txt')
DECAY        = args.decay
SUBDIVISION  = 2
BURN_IN      = 1000
lr_steps     = [[400000, 0.1], [450000, 0.1]]
weights_path = args.weights
NUM_CLASSES  = args.num_classes
IMG_SIZE     = 416
TRANS        = args.trans           # Transfer learning
FINETUNE     = args.finetune        # Fine tuning
SEP          = True if args.model == "sep" else False
restart      = args.restart
dropout      = args.dropout

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

# Load training dataloader
dataloader = _create_data_loader(
    TRAIN_PATH,
    BATCH_SIZE,
    IMG_SIZE
    )

# Load validation dataloader
validation_dataloader = _create_validation_data_loader(
    VALID_PATH,
    BATCH_SIZE,
    IMG_SIZE
    )

# Create model
if TRANS:
    model, param_to_update  = load_model(weights_path, device, NUM_CLASSES,
                                         trans=TRANS, restart=restart,
                                         finetune=FINETUNE, use_sep=SEP,
                                         dropout=dropout)
else:
    model = load_model(weights_path, device, NUM_CLASSES, tiny=True, trans=TRANS,
                       finetune=FINETUNE, use_sep=SEP, dropout=dropout)

# Define training algorithm and loss function
if TRANS:
    optimizer = optim.Adam(param_to_update, lr=LR, weight_decay=DECAY)
else:
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=DECAY)

print("Model :", model.__class__.__name__);

# Lists for loss graphs
losses  = []
mAPs    = []
max_map = 0
lrs     = []
lr      = LR

print("Start Training\n")
start = time.ctime()
start_cnv = time.strptime(start)
batches_done = 0
lr = LR

print(model)

#===============================================================================
# Training loop
#===============================================================================
for epoch in range(EPOCHS):
    model.train()
    for ite, (_, image, target) in enumerate(dataloader):
        batches_done = len(dataloader) * epoch + ite

        image = image.to(device)
        target = target.to(device)

        # Forward
        outputs = model(image)

        # Calculate loss
        loss, _ = compute_loss(outputs, target, model)

        # Backward
        loss.backward()

        # Adjust learning rate and run optimizer
        if batches_done % SUBDIVISION == 0:
            lr = LR
            if batches_done < BURN_IN:
                lr *= (batches_done / BURN_IN)
            else:
                for threshold, value in lr_steps:
                    if batches_done > threshold:
                        lr *= value

            for g in optimizer.param_groups:
                g['lr'] = lr

            optimizer.step()

            optimizer.zero_grad()

        #=======================================================================
        #optimizer.step()
        #optimizer.zero_grad()
        #=======================================================================

        #if batches_done % 10 == 0:
        print("\r[%3d][%d] Epoch / [%4d][%d] : loss = %.4f"
                % (epoch, EPOCHS, ite, len(dataloader), loss), end="")

    #=======================================================================
    #if epoch % 10 == 9:
    #    lr = lr * 0.1
    #    for g in optimizer.param_groups:
    #        g['lr'] = lr
    #=======================================================================

    # Validate on validation dataset
    if not args.novalid:
        sample_metrics = []
        labels         = []
        model.eval()
        print("Validating ...")
        for _, image, target in tqdm.tqdm(validation_dataloader):
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
        if max_map < mAP:
            max_map = mAP

        losses.append(loss.item())
        mAPs.append(mAP.item())
        lrs.append(lr)

        # save weight temp file
        torch.save(model.state_dict(), ".tmp_yolo_tiny.pt")

end = time.ctime()
end_cnv = time.strptime(end)

print("Start date >", time.strftime("%Y/%m/%d %H:%M:%S", start_cnv))
print("End date >", time.strftime("%Y/%m/%d %H:%M:%S", end_cnv))
# End training

#===============================================================================
# Save training result
#===============================================================================
if not args.nosave:
    # Prepare save location for training result and log
    result_dir = time.strftime("%Y%m%d_%H%M%S", start_cnv)
    result_path = os.path.join('results', result_dir) 
    os.makedirs(result_path)

    train_params_file = os.path.join(result_path, 'train_params.txt')
    with open(train_params_file, 'w') as f:
        f.write("Model : " + str(model.__class__.__name__) + "\n")
        f.write("epochs : " + str(EPOCHS) + "\n")
        f.write("batch_size : " + str(BATCH_SIZE) + "\n")
        f.write("pretrained_weights : " + str(args.weights) + "\n")
        f.write("learning_late : " + str(LR) + "\n")
        f.write("weight_decay : " + str(DECAY) + "\n")
        f.write("subdivision : " + str(SUBDIVISION) + "\n")
        f.write("burn_in : " + str(BURN_IN) + "\n")
        f.write("train_data_list : " + str(TRAIN_PATH) + "\n")
        f.write("num_classes : " + str(NUM_CLASSES) + "\n")
        f.write("image_size : " + str(IMG_SIZE) + "\n")
        f.write("trans : " + str(TRANS) + "\n")
        f.write("finetune : " + str(FINETUNE) + "\n")
        f.write("loss (last) :" + str(losses[-1]) + "\n")
        f.write("anchors :" + str(model.anchors) + "\n")
        f.write("MAX mAP :" + str(max_map) + "\n")
        f.write("restart :" + str(restart) + "\n")
        f.write("dropout :" + str(dropout) + "\n")
        f.write("\n")

    # Save model
    torch.save(model.state_dict(), os.path.join(result_path, args.output_model))

    # Create loss graph
    plot_graph(losses, EPOCHS, result_path + '/loss.png')
    plot_graph(mAPs, EPOCHS, result_path + '/mAP.png', label="mAP")
    plot_graph(lrs, EPOCHS, result_path + '/lr.png', label="learning rate")
