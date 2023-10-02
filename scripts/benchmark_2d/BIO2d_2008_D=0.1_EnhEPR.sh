CUDA_VISIBLE_DEVICES=7 python main.py \
    --seed 0 \
    --x_max 8. \
    --dimension 2 \
    --D 0.1 \
    --sample_size 10000 \
    --batch_size 2048 \
    --rho_1 0.1 \
    --rho_2 1. \
    --lr 0.001 \
    --resimulate \
    --boundary_type reflect \
    --model_save_dir checkpoints/BIO2DLC \
    --enh_scale 1. \
    --prefix enh_gauss  \
    --num_potential_epochs 5001 \
    --log_interval 1000 \
    --save_interval 1000 \
    --problem 2008 \
    --hidden_sizes 20,20,20 \
    --save_ckpt \
    --use_gpu

