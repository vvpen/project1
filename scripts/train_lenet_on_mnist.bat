

set TRAIN_DIR=D:\temp\ai\lenet-model


set DATASET_DIR=D:\temp\ai\mnist


python download_and_convert_data.py ^
  --dataset_name=mnist ^
  --dataset_dir=%DATASET_DIR%


python train_image_classifier.py ^
  --train_dir=%TRAIN_DIR% ^
  --dataset_name=mnist ^
  --dataset_split_name=train ^
  --dataset_dir=%DATASET_DIR% ^
  --model_name=lenet ^
  --preprocessing_name=lenet ^
  --max_number_of_steps=20000 ^
  --batch_size=50 ^
  --save_interval_secs=120 ^
  --save_summaries_secs=120 ^
  --log_every_n_steps=100 ^
  --optimizer=sgd ^
  --learning_rate_decay_type=fixed ^
  --learning_rate=0.01 ^
  --num_epochs_per_decay=200 ^
  --weight_decay=0



python eval_image_classifier.py ^
  --checkpoint_path=%TRAIN_DIR% ^
  --eval_dir=%TRAIN_DIR% ^
  --dataset_name=mnist ^
  --dataset_split_name=test ^
  --dataset_dir=%DATASET_DIR% ^
  --model_name=lenet
