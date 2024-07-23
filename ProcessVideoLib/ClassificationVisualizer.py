import io
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from matplotlib import pyplot as plt
#import requests
import cv2



class Visualizer(object):
  """Keeps track of claassifcations for every frame and renders them."""

  def __init__(self,
               class_name,
               plot_location_x=0.05,
               plot_location_y=0.05,
               plot_max_width=0.4,
               plot_max_height=0.4,
               plot_figsize=(9, 4),
               plot_x_max=None,
               plot_y_max=None,
               counter_location_x=0.65,
               counter_location_y=0.05,
               counter_font_path='Roboto-Regular.ttf',
               counter_font_color='red',
               counter_font_size=0.08):
    self._class_name = class_name
    self._plot_location_x = plot_location_x
    self._plot_location_y = plot_location_y
    self._plot_max_width = plot_max_width
    self._plot_max_height = plot_max_height
    self._plot_figsize = plot_figsize
    self._plot_x_max = plot_x_max
    self._plot_y_max = plot_y_max
    self._counter_location_x = counter_location_x
    self._counter_location_y = counter_location_y
    self._counter_font_path = counter_font_path
    self._counter_font_color = counter_font_color
    self._counter_font_size = counter_font_size

    self._counter_font = None

    self._pose_classification_history = []
    self._pose_classification_filtered_history = []
    
    #Plot angle components
    self._bias_x = 0.02
    self._bias_y = 0.02

    self._angle = dict()
    #self._image = None
    self._landmarks = None

    self._RightPartColor = (255,255,0)
    self._LeftPartColor = (100,150,255)

  def __call__(self,
               frame,
               pose_classification,
               pose_classification_filtered,
               repetitions_count,
               landmarks,
               angles):
    """Renders pose classifcation and counter until given frame."""
    # Extend classification history.
    self._pose_classification_history.append(pose_classification)
    self._pose_classification_filtered_history.append(pose_classification_filtered)

    self._angles = angles
    self._landmarks = landmarks
    
    frame = self._plot_angle(frame)
    # Output frame with classification plot and counter.
    output_img = Image.fromarray(frame)

    output_width = output_img.size[0]
    output_height = output_img.size[1]

    # Draw the plot.
    img = self._plot_classification_history(output_width, output_height)
    img.thumbnail((int(output_width * self._plot_max_width),
                   int(output_height * self._plot_max_height)),
                  Image.ANTIALIAS)
    output_img.paste(img,
                     (int(output_width * self._plot_location_x),
                      int(output_height * self._plot_location_y)))

    # Draw the count.
    output_img_draw = ImageDraw.Draw(output_img)
    if self._counter_font is None:
      font_size = int(output_height * self._counter_font_size)
      #font_request = requests.get(self._counter_font_path, allow_redirects=True)
      #self._counter_font = ImageFont.truetype(io.BytesIO(font_request.content), size=font_size)
      font_file = open(self._counter_font_path,'rb')
      font_request = font_file.read()
      font_file.close()
      self._counter_font = ImageFont.truetype(io.BytesIO(font_request), size=font_size)
    output_img_draw.text((output_width * self._counter_location_x,
                          output_height * self._counter_location_y),
                         str(repetitions_count),
                         font=self._counter_font,
                         fill=self._counter_font_color)

    return output_img

  def _plot_angle(self, frame):
    
    image_hight, image_width, _ = frame.shape
    
    for center, angle in self._angles.items():
      location_x = (self._landmarks[center].x - self._bias_x) * image_width
      location_y = (self._landmarks[center].y + self._bias_y) * image_hight
      if angle is None:
        continue

      if "RIGHT" in str(center):
        cv2.putText(frame, '{:.2f}'.format(angle), (int(location_x), int(location_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self._RightPartColor, 2)
      else:
        cv2.putText(frame, '{:.2f}'.format(angle), (int(location_x), int(location_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self._LeftPartColor, 2)

      return frame

  def _plot_classification_history(self, output_width, output_height):
    fig = plt.figure(figsize=self._plot_figsize)

    for classification_history in [self._pose_classification_history,
                                   self._pose_classification_filtered_history]:
      y = []
      for classification in classification_history:
        if classification is None:
          y.append(None)
        elif self._class_name in classification:
          y.append(classification[self._class_name])
        else:
          y.append(0)
      plt.plot(y, linewidth=7)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Frame')
    plt.ylabel('Confidence')
    plt.title('Classification history for `{}`'.format(self._class_name))
    plt.legend(loc='upper right')

    if self._plot_y_max is not None:
      plt.ylim(top=self._plot_y_max)
    if self._plot_x_max is not None:
      plt.xlim(right=self._plot_x_max)

    # Convert plot to image.
    buf = io.BytesIO()
    dpi = min(
        output_width * self._plot_max_width / float(self._plot_figsize[0]),
        output_height * self._plot_max_height / float(self._plot_figsize[1]))
    fig.savefig(buf, dpi=dpi)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()

    return img
