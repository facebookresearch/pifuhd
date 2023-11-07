# Multi-view PIFuHD Using the Single-view Pretrained Model

<p align="center">
  <img src="diagram.png" width="500" height="500">
</p>

[PIFuHD](https://shunsukesaito.github.io/PIFuHD/) only provides a pretrained model for single-view (frontal) image 3D reconstruction. This repository extends its functionality to support 3D reconstruction from multi-view (frontal/backside) images, employing the same pretrained single-view model. "Zeroshot" multi-view 3D reconstruction is achievable by modifying the image processing pipelines and applying channel-wise Adaptive Instance Normalization (AdaIN). This multi-view approach enhances the detail on the backside of the 3D mesh.

## Setup
Please refer to the original [PIFuHD](https://shunsukesaito.github.io/PIFuHD/) repository for setup instructions, which include package installation and downloading the pretrained model.

## Preprocessing
For 3D reconstruction, a {frontal_image_name}_rect.txt file is required. If the input images are center-aligned, a .txt file with the contents '0 0 {image_height} {image_width}' should suffice.
If the input images are not centered-aligned, refer to the original [Testing section](https://shunsukesaito.github.io/PIFuHD/) to obtain `{frontal_image_name}_rect.txt`. 

## Testing
Run the following command for testing:
```
python -m apps.simple_test -i {folder_containing_images_and_rect} -b {path_of_backside_image} --use_rect 
```

## Modification
Major code modifications: 
```
    .
    ├── apps
    |   ├── simple_test.py         # command line argument for a backside image 
    |   └── recon.py               # feed a backside image to PIFuHD 
    └── lib                        
        ├── model                  
        |   └── HGPIFuNetwNML.py   # obtain a backside normal map and apply AdaIN.
        └── data                   
            └── EvalDataset.py     # image preprocessing for a backside image
    
```
    
## License
[CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode). 
See the [LICENSE](LICENSE) file. 
