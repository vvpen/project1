

rem Where the checkpoint and logs will be saved to.
set TRAIN_DIR=D:\temp\ai\densenet-train

rem Where the dataset is saved to.
set DATASET_DIR=D:\temp\ai\cifar10

rem Download the dataset
python download_and_convert_data.py ^
  --dataset_name=cifar10 ^
  --dataset_dir=%DATASET_DIR%

rem Run training.
python train_image_classifier.py ^
  --train_dir=%TRAIN_DIR% ^
  --dataset_name=cifar10 ^
  --dataset_split_name=train ^
  --dataset_dir=%DATASET_DIR% ^
  --model_name=densenet ^
  --preprocessing_name=vgg ^
  --max_number_of_steps=500 ^
  --batch_size=100 ^
  --save_interval_secs=120 ^
  --save_summaries_secs=120 ^
  --log_every_n_steps=100 ^
  --optimizer=adam ^
  --learning_rate=0.1 ^
  --learning_rate_decay_factor=0.1 ^
  --num_epochs_per_decay=200 ^
  --weight_decay=0.004

rem Run evaluation.
python eval_image_classifier.py ^
  --checkpoint_path=%TRAIN_DIR% ^
  --eval_dir=%TRAIN_DIR% ^
  --dataset_name=cifar10 ^
  --dataset_split_name=test ^
  --dataset_dir=%DATASET_DIR% ^
  --model_name=densenet ^
  --preprocessing_name=vgg ^