import pybullet_data as pd


class Scene(object):
    """Class of the scene"""

    def __init__(self, pybullet_client):
        
        self.pybullet_client = pybullet_client
        self.create_scene()
    
    
    def create_scene(self):
        pass
        