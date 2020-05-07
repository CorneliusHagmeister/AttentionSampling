
python ./infer.py --experiment_names 100p/DCGAN/imagenet-192 --checkpoint_name="Epoch_inter(50).ckpt" --output_dir=generated_imgs_ep50 --num_samples=5000
python ./infer.py --experiment_names 100p/DRAGAN/imagenet-192 --checkpoint_name="Epoch_inter(50).ckpt"  --output_dir=generated_imgs_ep50 --num_samples=5000
python ./infer.py --experiment_names 100p/LSGAN/imagenet-192 --checkpoint_name="Epoch_inter(50).ckpt"  --output_dir=generated_imgs_ep50 --num_samples=5000
python ./infer.py --experiment_names 100p/WGANGP/imagenet-192 --checkpoint_name="Epoch_inter(50).ckpt" --output_dir=generated_imgs_ep50 --num_samples=5000

