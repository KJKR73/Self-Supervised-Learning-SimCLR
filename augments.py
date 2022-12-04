import numpy as np

class AugmentWithViews(object):
    '''Makes the simclr dataset'''
    def __init__(self, transforms, views):
        # Make the instance variables
        self.views = views
        self.transforms = transforms
        
    def __call__(self, image):
        return [self.transforms(image=np.array(image))['image'] for i in range(self.views)]


class SimpleAugment(object):
    '''Makes the simclr dataset'''
    def __init__(self, transforms):
        # Make the instance variables
        self.transforms = transforms
        
    def __call__(self, image):
        return self.transforms(image=np.array(image))['image']