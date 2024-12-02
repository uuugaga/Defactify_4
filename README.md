# Defactify 4 AAAI 2025 Workshop

# Train model
When training model with prompt `--noise_sigma x`, means add noise with sigma from 0 to x randomly.
```
python classifier.py --features_selected rgb
python classifier.py --features_selected rgb --noise_sigma 0.2
```

# Generate validation result
```
python classifier.py --features_selected rgb --evaluate
```

# Use Grad-cam
```
python classifier.py --features_selected rgb --grad_cam 
```

# Inference
```
python classifier.py --features_selected rgb --inference
python classifier.py --features_selected rgb --inference --labels_file ../data/test/captions.xlsx
```

## feature_extraction with Robustness test  
```
python feature_extraction.py --data_path ../data/val --results_path ../data/val_compression --error --frequency --flat_structure --compression_quality 80
python feature_extraction.py --data_path ../data/val --results_path ../data/val_crop --error --frequency --flat_structure --crop_factor 0.8
python feature_extraction.py --data_path ../data/val --results_path ../data/val_blur --error --frequency --flat_structure --blur_sigma 2
python feature_extraction.py --data_path ../data/val --results_path ../data/val_noise --error --frequency --flat_structure --noise_sigma 0.1
```

