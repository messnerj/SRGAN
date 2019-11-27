for img_loss in 1 0.5
do
    for adv_loss in 4e-2 2e-2 1e-2
    do
        for percept_loss in 1e-1 6e-2 3e-2
        do
            echo "img_loss: $img_loss, adv_loss: $adv_loss, percept_loss: $percept_loss"
            CUDA_VISIBLE_DEVICES=1 python train.py --epochs 100 --img_loss $img_loss --adv_loss $adv_loss --percept_loss $percept_loss
        done
    done
done

