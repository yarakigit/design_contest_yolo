NUM_CLASSES	  = 2
CLASS_FILE	  = namefiles/contest_2cls.names
DETECT_OPTION     = --num_classes $(NUM_CLASSES) --class_names $(CLASS_FILE)
DETECT_IMAGE	  = sample/test_doll_light.png
WEIGHTS	          = sample/doll_light_tiny.pt 
EPOCHS            = 50
DATA-ROOT         = /data/design_contest_dataset/contest_doll_light

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

train_continue_sep:
	 python train.py --weights $(WEIGHTS) $(DETECT_OPTION) --epochs $(EPOCHS) --data_root $(DATA-ROOT)--model sep

detect_sep: 
	python detect.py --weights $(WEIGHTS) $(DETECT_OPTION) --image $(DETECT_IMAGE) --mode sep

cal_mAP_sep:
        python cal_mAP.py --weights $(WEIGHTS) $(DETECT_OPTION) --data_root $(DATA-ROOT) --mode sep
