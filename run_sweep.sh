for lr_g in 1e-3 3e-4 1e-4 
do
    for lr_d in 1e-3 3e-4 1e-4
    do
        for img_loss in 1 1e-1 1e-2
        do
            for adv_loss in 1 1e-2 1e-2
            do
                echo "lr_g: $lr_g, lr_d: $lr_d, img_loss: $img_loss, adv_loss: $adv_loss"
                CUDA_VISIBLE_DEVICES=1 python train.py --epochs 100 --lr_g $lr_g --lr_d $lr_d --img_loss $img_loss --adv_loss $adv_loss
            done
        done
    done
done
