class ImageBundle:
    def __init__(self, img, name, meta):
        self.img = img
        self.name = name
        self.meta = meta
        self.img_512 = None
        self.calib = None
        self.calib_world = None
        self.b_min = None
        self.b_max = None
        self.is_multi = False
