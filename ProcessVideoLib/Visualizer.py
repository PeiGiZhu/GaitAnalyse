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
               plot_figsize=(9, 5),
               plot_x_max=None,
               plot_y_max=None,
               counter_location_x=0.80,
               counter_location_y=0.05,
               speed_indicator_location_x=0.70,
               speed_indicator_location_y=0.15,
               counter_font_path='Roboto-Regular.ttf',
               counter_font_color='red',
               counter_font_size=0.03):
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
    self._speed_indicator_location_x = speed_indicator_location_x
    self._speed_indicator_location_y = speed_indicator_location_y
    self._counter_font_path = counter_font_path
    self._counter_font_color = counter_font_color
    self._counter_font_size = counter_font_size

    self._counter_font = None
    '''
    self._pose_classification_history = []
    self._pose_classification_filtered_history = []
    '''
    #Plot angle components
    self._char_bias_x = 0.02
    self._char_bias_y = 0.02
    self._plot_bias_mul = 8

    self._angles_history = []
    self._frame = None
    self._landmarks = dict()

    self._RightPartColor = (255,255,0)
    self._LeftPartColor = (100,150,255)

  def __call__(self,
               frame,
               pose_classification,
               pose_classification_filtered,
               repetitions_count,
               stride_speed,
               landmarks,
               angles):
    """Renders pose classifcation and counter until given frame."""
    self._angles_history.append(angles)
    self._landmarks = landmarks
    self._frame = frame

    '''
    # Extend classification history.
    self._pose_classification_history.append(pose_classification)
    self._pose_classification_filtered_history.append(pose_classification_filtered)
    '''
    if landmarks is not None:
        self._plot_angle(angles)
    
    # print("Look here: ", self._angles)
    # Output frame with classification plot and counter.
    output_img = Image.fromarray(self._frame)

    output_width = output_img.size[0]
    output_height = output_img.size[1]

    # Draw the plot.
    img = self._plot_parameter_history(output_width, output_height)
      
    img.thumbnail((int(output_width * self._plot_max_width),
                    int(output_height * self._plot_max_height)),
                    Image.ANTIALIAS)
      
    output_img.paste(img,
                     (int(output_width * self._plot_location_x),
                      int(output_height * self._plot_location_y)))
      


    # Draw the count.
    output_img_draw = ImageDraw.Draw(output_img)
    if self._counter_font is None:
      font_size = int(output_width * self._counter_font_size)
      #font_request = requests.get(self._counter_font_path, allow_redirects=True)
      #self._counter_font = ImageFont.truetype(io.BytesIO(font_request.content), size=font_size)
      font_file = open(self._counter_font_path,'rb')
      font_request = font_file.read()
      font_file.close()
      self._counter_font = ImageFont.truetype(io.BytesIO(font_request), size=font_size)
    output_img_draw.text((output_width * self._counter_location_x,
                          output_height * self._counter_location_y),
                         "stride(s): " + str(repetitions_count),
                         font=self._counter_font,
                         fill=self._counter_font_color)
    
    output_img_draw.text((output_width * self._speed_indicator_location_x,
                          output_height * self._speed_indicator_location_y),
                         '{:.2f}'.format(stride_speed*60) + " strides / min",
                         font=self._counter_font,
                         fill=self._counter_font_color)
    
    return output_img

  def _plot_angle(self, angles):
    
    image_hight, image_width, _ = self._frame.shape
    
    for center, angle in angles.items():
      #Angle return from JointAngle module may contain None value
      #Angle return from Kalmanfilter module may contain 0 value
      #The 0 value came from detection failed(None value from JointAngle module),
      #and the kalmanfilter provide 0 statement value initially
      if angle is None or angle == 0:
        continue
      #print("-- Here -- ",center, angle)
      location_x = (self._landmarks[center].x - self._char_bias_x) * image_width
      location_y = (self._landmarks[center].y + self._char_bias_y) * image_hight

      if "RIGHT" in str(center):
        cv2.putText(self._frame, '{:.2f}'.format(angle), (int(location_x), int(location_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self._RightPartColor, 2)
      else:
        cv2.putText(self._frame, '{:.2f}'.format(angle), (int(location_x), int(location_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self._LeftPartColor, 2)


  def _plot_parameter_history(self, output_width, output_height):

    fig = plt.figure(1, figsize=self._plot_figsize)
    '''
    #plot classification history
    plt.subplot(2,1,1)
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
    '''
    #plot hip angle
    plt.subplot(2,1,1)
    
    left_hip = []
    right_hip = []
    left_detected = 'left hip'
    right_detected = 'right hip'
    
    for angle in self._angles_history:
      get_left_hip = 0
      get_right_hip = 0

      for center, angle in angle.items():
        if "LEFT_HIP" in str(center):
          left_hip.append(angle)
          get_left_hip = 1
          if angle > 160 or angle < -20:
            left_detected = 'left hip failed'
            
        if "RIGHT_HIP" in str(center):
          right_hip.append(angle)
          get_right_hip = 1
          if angle > 160 or angle <-20:
            right_detected = 'right hip failed'

      if get_left_hip == 0:
        left_hip.append(None)
      if get_right_hip == 0:
        right_hip.append(None)

    left_hip_line,  = plt.plot(left_hip, linewidth = 7)
    right_hip_line,  = plt.plot(right_hip, linewidth = 7)

    plt.grid(axis='y', alpha=0.75)
    plt.ylabel('Hip Angle degree °')
    plt.legend( (left_hip_line, right_hip_line),
                     [left_detected, right_detected],
                     loc='upper right' )

    if self._plot_y_max is not None:
      plt.ylim(top= 150, bottom = 30)
    if self._plot_x_max is not None:
      plt.xlim(left = 0, right=self._plot_x_max)
      
    #plot knee angle
    plt.subplot(2,1,2)

    left_knee = []
    right_knee = []
    left_detected = 'left knee'
    right_detected = 'right knee'
    
    for angle in self._angles_history:
      get_left_knee = 0
      get_right_knee = 0
      for center, angle in angle.items():
        if "LEFT_KNEE" in str(center):
          left_knee.append(angle)
          get_left_knee = 1
          if angle > 200 or angle < 70:
            left_detected = 'left knee failed'
        if "RIGHT_KNEE" in str(center):
          right_knee.append(angle)
          get_right_knee = 1
          if angle > 200 or angle < 70:
            right_detected = 'right knee failed'

      if get_left_knee == 0:
        left_knee.append(None)
      if get_right_knee == 0:
        right_knee.append(None)
            
    left_knee_line, = plt.plot(left_knee, linewidth = 7)  
    right_knee_line, = plt.plot(right_knee, linewidth = 7)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Frame')
    plt.ylabel('Knee Angle degree °')
    plt.legend( (left_knee_line, right_knee_line),
                     [left_detected, right_detected],
                     loc='upper right' )

    if self._plot_y_max is not None:
      plt.ylim(top= 200, bottom = 90)
    if self._plot_x_max is not None:
      plt.xlim(left = 0, right=self._plot_x_max)
    
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
