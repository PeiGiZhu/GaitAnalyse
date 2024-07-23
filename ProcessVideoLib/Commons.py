from matplotlib import pyplot as plt

class ShowImage(object):
  """Shows output PIL image."""
  
  def __init__(self,
              figsize=(10,10)):
    self._figsize = figsize
  def __call__(self, img):
    plt.figure(figsize=self._figsize)
    plt.imshow(img)
    plt.show()

'''
def show_image(img, figsize=(10, 10)):
  """Shows output PIL image."""
  plt.figure(figsize=figsize)
  plt.imshow(img)
  plt.show()
'''
