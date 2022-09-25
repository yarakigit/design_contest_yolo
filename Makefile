NUM_CLASSES	  = 2
CLASS_FILE	  = namefiles/contest_2cls.names
DETECT_OPTION     = --num_classes $(NUM_CLASSES) --class_names $(CLASS_FILE)
DETECT_IMAGE	  = sample/test_doll_light.png
WEIGHTS	          = sample/doll_light_tiny.pt 
EPOCHS            = 50
DATA-ROOT         = /data/design_contest_dataset/contest_doll_light

## For cal_mAP_external_file
MAP_IMAGE_DIR    = /your-custom-dir
MAP_IMAGE_DIR_OUT = ./output_for_mAP
FILE_TYPE = jpg   # OID-defalut : jpg 
CONF_THRES = 0.1  # DEFAULT : 0.1
NMS_THRES  = 0.4  # DEFAULT : 0.4


train:
	python train.py $(DETECT_OPTION) --epochs $(EPOCHS) --data_root $(DATA-ROOT)

train_continue:
	python train.py --weights $(WEIGHTS) $(DETECT_OPTION) --epochs $(EPOCHS) --data_root $(DATA-ROOT)

detect: 
	python detect.py --weights $(WEIGHTS) $(DETECT_OPTION) --image $(DETECT_IMAGE)

cal_mAP:
	python cal_mAP.py --weights $(WEIGHTS) $(DETECT_OPTION) --data_root $(DATA-ROOT)

train_sep:
	python train.py $(DETECT_OPTION) --epochs $(EPOCHS) --data_root $(DATA-ROOT) --model sep

cal_mAP_external_file:cal_mAP_external_file_pre
	python cal_mAP_external_file.py --weights $(WEIGHTS) $(DETECT_OPTION) --map_image_dir $(MAP_IMAGE_DIR) --map_image_dir_out $(MAP_IMAGE_DIR_OUT) --file_type $(FILE_TYPE) --conf_thres $(CONF_THRES) --nms_thres $(NMS_THRES) --nogpu

cal_mAP_external_file_vis:cal_mAP_external_file_pre
	python cal_mAP_external_file.py --weights $(WEIGHTS) $(DETECT_OPTION) --map_image_dir $(MAP_IMAGE_DIR) --map_image_dir_out $(MAP_IMAGE_DIR_OUT) --file_type $(FILE_TYPE) --conf_thres $(CONF_THRES) --nms_thres $(NMS_THRES) --nogpu --nms_vis_en

##
cal_mAP_external_file_pre:
	rm -rf $(MAP_IMAGE_DIR_OUT)
	mkdir $(MAP_IMAGE_DIR_OUT)
	mkdir $(MAP_IMAGE_DIR_OUT)/images
	mkdir $(MAP_IMAGE_DIR_OUT)/labels

##  For sep
train_continue_sep:
	python train.py --weights $(WEIGHTS) $(DETECT_OPTION) --epochs $(EPOCHS) --data_root $(DATA-ROOT)--model sep

detect_sep: 
	python detect.py --weights $(WEIGHTS) $(DETECT_OPTION) --image $(DETECT_IMAGE) --mode sep

cal_mAP_sep:
	python cal_mAP.py --weights $(WEIGHTS) $(DETECT_OPTION) --data_root $(DATA-ROOT) --mode sep

cal_mAP_external_file_sep:cal_mAP_external_file_pre
	python cal_mAP_external_file.py --weights $(WEIGHTS) $(DETECT_OPTION) --map_image_dir $(MAP_IMAGE_DIR) --map_image_dir_out $(MAP_IMAGE_DIR_OUT) --file_type $(FILE_TYPE) --conf_thres $(CONF_THRES) --nms_thres $(NMS_THRES) --nogpu --model sep

cal_mAP_external_file_vis_sep:cal_mAP_external_file_pre
	python cal_mAP_external_file.py --weights $(WEIGHTS) $(DETECT_OPTION) --map_image_dir $(MAP_IMAGE_DIR) --map_image_dir_out $(MAP_IMAGE_DIR_OUT) --file_type $(FILE_TYPE) --conf_thres $(CONF_THRES) --nms_thres $(NMS_THRES) --nogpu --nms_vis_en --model sep
