MODE=test
VBS=1
BS=1

CHECKPOINT=400
CHECKPOINT_DIR=/home/andy/Dropbox/largefiles1/logs/cut_bs8/checkpoints
VALDIR="/home/andy/Dropbox/largefiles1/autoferry_processed/autoferry"
EVAL_DIR="inferences"

python eval.py \
--dataroot $VALDIR \
--checkpoints_dir $CHECKPOINT_DIR \
--results_dir $EVAL_DIR \
--name autoferry_cut \
--preprocess resize \
--model cut \
--epoch $CHECKPOINT \
--input_nc 3 \
--output_nc 3 \
--batch_size 8 \
--phase test \
--no_dropout true \
--load_size 256 \
--crop_size 255 \
--num_threads 4

# /home/andy/Dropbox/largefiles1/logs/cut_bs1/checkpoints/autoferry_cut
# /home/andy/Dropbox/largefiles1/logs/cut_bs8/checkpoints/autoferry_cut/285_net_G.pth