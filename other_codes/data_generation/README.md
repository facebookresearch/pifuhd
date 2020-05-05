## Requirements
- Python 3
- trimesh
- OpenGL
- cv2
- pyexr
    - how to install in fedora:
        - sudo dnf install openexr
        - sudo dnf install openexr-devel
        - pip install pyexr        

## Data Generation
1. run the following command after setting paths in ./apps/generate_pifu.py
```
python -m apps.generate_pifu
```