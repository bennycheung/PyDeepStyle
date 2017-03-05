#

if [ "$#" -ne 3 ]; then
  echo "need to specify <content_image> <style_image> <output_dir>"
  exit
fi

IMAGE_PATH=images

mkdir -p ${IMAGE_PATH}/$3

python deepstyle.py "${IMAGE_PATH}/$1" "${IMAGE_PATH}/$2" "${IMAGE_PATH}/$3/$3" \
  --image 200 \
  --content_weight 0.025 --style_weight 1.0 \
  --total_variation_weight 8.5E-05 --style_scale 1 \
  --num_iter 25 --rescale_image "False" --rescale_method "bicubic" \
  --maintain_aspect_ratio "True" --content_layer "conv5_2" \
  --init_image "content" --pool_type "max" \
  --preserve_color "False" --min_improvement 0 --model "vgg16" --content_loss_type 0

