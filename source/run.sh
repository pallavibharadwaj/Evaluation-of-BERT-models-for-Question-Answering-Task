rm -r cache_dir
python train.py distilbert

rm -r cache_dir
python train.py roberta

rm -r cache_dir
python train.py electra-base