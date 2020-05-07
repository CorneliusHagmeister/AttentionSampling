base_path="../DCGAN/output_new/output/100p/"

echo "${base_path}cifar_10_0"

# compute eval metrics for cifar10 100p
python ./compute_eval_scores.py --base="${base_path}DCGAN/cifar10" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9

 python ./compute_eval_scores.py --base="${base_path}DCGAN/cifar10-botk" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9

python ./compute_eval_scores.py --base="${base_path}DCGAN/cifar10-topk" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9

python ./compute_eval_scores.py --base="${base_path}DCGAN/cifar10-wrs" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9

#  LSGAN
python ./compute_eval_scores.py --base="${base_path}LSGAN/cifar10" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9

 python ./compute_eval_scores.py --base="${base_path}LSGAN/cifar10-botk" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9

python ./compute_eval_scores.py --base="${base_path}LSGAN/cifar10-topk" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9

python ./compute_eval_scores.py --base="${base_path}LSGAN/cifar10-wrs" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9

#DRAGAN
python ./compute_eval_scores.py --base="${base_path}DRAGAN/cifar10" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9

 python ./compute_eval_scores.py --base="${base_path}DRAGAN/cifar10-botk" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9

python ./compute_eval_scores.py --base="${base_path}DRAGAN/cifar10-topk" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9

python ./compute_eval_scores.py --base="${base_path}DRAGAN/cifar10-wrs" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9

 #DRAGAN
python ./compute_eval_scores.py --base="${base_path}WGANGP/cifar10" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9

 python ./compute_eval_scores.py --base="${base_path}WGANGP/cifar10-botk" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9

python ./compute_eval_scores.py --base="${base_path}WGANGP/cifar10-topk" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9

python ./compute_eval_scores.py --base="${base_path}WGANGP/cifar10-wrs" --save_folder=eval_ep100 --inception_filename_base=inception-v3_ep100.npy --compare \
../DCGAN/output/cifar_base/full ../DCGAN/output/cifar_base/0  ../DCGAN/output/cifar_base/1 ../DCGAN/output/cifar_base/2 ../DCGAN/output/cifar_base/3 ../DCGAN/output/cifar_base/4 ../DCGAN/output/cifar_base/5 ../DCGAN/output/cifar_base/6 ../DCGAN/output/cifar_base/7 ../DCGAN/output/cifar_base/8 ../DCGAN/output/cifar_base/9 \
 --labels cifar_full cifar_base_0 cifar_base_1 cifar_base_2 cifar_base_3 cifar_base_4 cifar_base_5 cifar_base_6 cifar_base_7 cifar_base_8 cifar_base_9