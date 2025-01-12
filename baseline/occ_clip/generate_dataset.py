import os
import json
import random
import pandas as pd

if __name__ == "__main__":

    entry = "../../data/train"
    
    fake_names = [
        "sd21", "sdxl", "sd3",
        "dalle", "midjourney",
    ]
    
    ################
    #  generate the training data
    #####################################
    non_target_data = [
        os.path.join(entry, "coco_image", filename) 
        for filename in os.listdir(os.path.join(entry, "coco_image")) if "png" in filename
    ]

    data = {
        "train": []
    }
    for i, target_name in enumerate(fake_names):
  
        new_entry = os.path.join(entry, f"{target_name}_image")
        target_data = [
            os.path.join(new_entry, filename) for filename in os.listdir(new_entry) if "png" in filename
        ]
        
        data["train"] = non_target_data + target_data              

        with open(f"dataset/TrainData{i+1}.json", "w") as out_file:
            json.dump(data, out_file, indent=4)

    ###################################################################
    # generate val data
    #############################################
    entry = "../../data/val"
    data = {
        "val": []
    }
    val_shuffle_df = pd.read_csv(os.path.join("..", "..", "data", "val", "val_shuffle.csv"))

    df = val_shuffle_df.groupby("Label_B").get_group(0)
    indices = df.index

    non_target_data = [
        os.path.join(entry, filename) 
        for filename in os.listdir(os.path.join(entry)) 
        if "png" in filename and int(filename.split(".")[0]) in indices
    ]

    for i, target_name in enumerate(fake_names):
        df = val_shuffle_df.groupby("Label_B").get_group(i+1)
        indices = df.index

        target_data = [
            os.path.join(entry, filename) 
            for filename in os.listdir(os.path.join(entry)) 
            if "png" in filename and int(filename.split(".")[0]) in indices
        ]
        data["val"] = non_target_data + target_data
        with open(f"dataset/ValData{i+1}.json", "w") as out_file:
            json.dump(data, out_file, indent=4)