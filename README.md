# Defactify 4 AAAI 2025 Workshop


# Generate prediction result
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
```

## feature_extraction with Robustness test  
```
python feature_extraction.py --data_path ../data/val --results_path ../data/val_compression --error --frequency --flat_structure --compression_quality 80
python feature_extraction.py --data_path ../data/val --results_path ../data/val_crop --error --frequency --flat_structure --crop_factor 0.8
python feature_extraction.py --data_path ../data/val --results_path ../data/val_blur --error --frequency --flat_structure --blur_sigma 2
python feature_extraction.py --data_path ../data/val --results_path ../data/val_noise --error --frequency --flat_structure --noise_sigma 0.1
```

