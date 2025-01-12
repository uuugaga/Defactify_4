# Reproduce the performance of the baseline methods

## AEROBLADE
- make sure you're in the `baseline/aeroblade` directory
    ```shell
    cd baseline/aeroblade
    ```
- directly run the `run.sh`, you can adjust the hyperparameters based on your hardware device
    ```shell
    bash run.sh
    ```
- the LPIPS-2 distance of all dataset will store in `baseline/aeroblade/experiment` directory, please run all code block in the `visualization.ipynb` and the final score will show in the notebook

## OCC-CLIP
- make sure you're in the `baseline/occ_clip` directory
    ```shell
    cd baseline/occ_clip
    ```
- generate the two-class classifier dataset 
    ```shell
    python generate_dataset.py
    ```
- directly run the `run.sh`, you can adjust the hyperparameters based on your hardware device
    ```shell
    bash run.sh
    ```
- each training result will store in `baseline/occ_clip/result` directory. run all code blocks in `aggregate.ipynb` to get the multi-class classification score from multiple binary classifiers.


## Acknowledgements
- [AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error (CVPR 2024)](https://github.com/jonasricker/aeroblade)
- [Which Model Generated This Image? A Model-Agnostic Approach for Origin Attribution (ECCV 2024)](https://github.com/uwFengyuan/OCC-CLIP)