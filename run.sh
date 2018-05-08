# Shell scrhipt to drive deepstyle.py
# Usage: ./run.sh <content_image> <style_image> <output_dir>
#


if [ "$#" -ne 3 ]; then
  echo "need to specify <content_image> <style_image> <output_dir>"
  exit
fi

mkdir -p $3

python neural_style_transfer.py "$1" "$2" "$3/$3" \
  --image_size 600 \
  --tv_weight 8.5E-05 \
  --iter 25 \
  --content_layer "block5_conv2" \
  --min_improvement 0
