#export CUDA_VISIBLE_DEVICES=1
python3 inference.py ./cfgs/custom.yaml \
--weight \path\to\saved_model.pdparams \
--content_font \path\to\content_imgs \
--img_path \path\to\test_imgs \
--saving_root ./infer_res


