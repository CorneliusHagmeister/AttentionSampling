# DCGAN
python ./train.py --epochs=25 --dataset=imagenet  --experiment_name=100p/DCGAN/imagenet-normal --adversarial_loss_mode=gan --top_k=False --normal=True --k=192

# DRAGAN
python ./train.py --epochs=25 --dataset=imagenet  --experiment_name=100p/DRAGAN/imagenet-normal --adversarial_loss_mode=gan --top_k=False --normal=True --k=192 --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=dragan

# LSGAN
python ./train.py --epochs=25 --dataset=imagenet  --experiment_name=100p/LSGAN/imagenet-normal --adversarial_loss_mode=lsgan --top_k=False --normal=True --k=192

# WGAN-GP
python ./train.py --epochs=25 --dataset=imagenet  --experiment_name=100p/WGANGP/imagenet-normal --adversarial_loss_mode=wgan --top_k=False --normal=True --k=192 --gradient_penalty_mode=1-gp --gradient_penalty_sample_mode=line --n_d=5
