python run_aeroblade.py --num-workers 4 --batch-size 16 --files-or-dirs ../../data/train/coco_image/ --output-dir experiment/train/coco_image
python run_aeroblade.py --num-workers 4 --batch-size 16 --files-or-dirs ../../data/train/dalle_image/ --output-dir experiment/train/dalle_image
python run_aeroblade.py --num-workers 4 --batch-size 16 --files-or-dirs ../../data/train/midjourney_image/ --output-dir experiment/train/midjourney_image
python run_aeroblade.py --num-workers 4 --batch-size 16 --files-or-dirs ../../data/train/sd3_image/ --output-dir experiment/train/sd3_image
python run_aeroblade.py --num-workers 4 --batch-size 16 --files-or-dirs ../../data/train/sd21_image/ --output-dir experiment/train/sd21_image
python run_aeroblade.py --num-workers 4 --batch-size 16 --files-or-dirs ../../data/train/sdxl_image/ --output-dir experiment/train/sdxl_image
python run_aeroblade.py --num-workers 4 --batch-size 16 --files-or-dirs ../../data/val/ --output-dir experiment/val/