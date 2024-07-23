
class LandMarks:
  def __init__(self, x, y, z, visibility):
    self._visibility = visibility
    self._x = x
    self._y = y
    self._z = z
  
  @property
  def visibility(self):
    return self._visibility
  
  @visibility.setter
  def visibility(self, value):
    self._visibility = value

  @property
  def x(self):
    return self._x

  @x.setter
  def x(self, value):
    self._x = value

  @property
  def y(self):
    return self._y
  
  @y.setter
  def y(self, value):
    self._y = value
  
  @property
  def z(self):
    return self._z
  
  @z.setter
  def z(self, value):
    self._z = value



class Landmarks:
  def __init__(self, landmarks):
    self._visibility = landmarks.visibility
    self._x = landmarks.x
    self._y = landmarks.y
    self._z = landmarks.z
  
  @property
  def visibility(self):
    return self._visibility
  
  @visibility.setter
  def visibility(self, value):
    self._visibility = value

  @property
  def x(self):
    return self._x

  @x.setter
  def x(self, value):
    self._x = value

  @property
  def y(self):
    return self._y
  
  @y.setter
  def y(self, value):
    self._y = value
  
  @property
  def z(self):
    return self._z
  
  @z.setter
  def z(self, value):
    self._z = value
