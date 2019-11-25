for img_loss in 1 1e-1 1e-2
do
    for adv_loss in 1 1e-1 1e-2
    do
        for percept_loss in 1e-1 3e-2 1e-2
        do
            echo "img_loss: $img_loss, adv_loss: $adv_loss, percept_loss: $percept_loss"
            CUDA_VISIBLE_DEVICES=1 python train.py --epochs 100 --img_loss $img_loss --adv_loss $adv_loss --percept_loss $percept_loss
        done
    done
done

