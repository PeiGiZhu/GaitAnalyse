class StrideSpeed(object):
    """Calculate the stride speed"""
    def __init__(self, fps, count_windows = 1):

        # need frames per second and stride counter
        self._fps = fps

        # every 'count_windows' strides, updata stride speed one time
        self._count_windows = count_windows

        self._last_frame_id = 0
        self._curr_frame_id = 0
        self._frame_interval = 0

        self._last_strides_num = 0
        self._curr_strides_num = 0

        self._updata_speed = False
        self._stride_speed = 0

    def __call__(self, stride_count, curr_frame_id):
        """
        the stride speed equal to
        count_windows * fps / frames interval.

        Args:
            stride_count: the number of stride from start frame to current frame
            curr_frame_id: the current frame id
        Return:
            self._stride_speed: the stride speed during last count windows
        """
        self._curr_strides_num = stride_count
        self._curr_frame_id = curr_frame_id
        
        if self._curr_strides_num - self._last_strides_num >= self._count_windows:
            self._last_strides_num = self._curr_strides_num
            self._updata_speed = True

        if self._updata_speed is True:
            self._frame_interval = self._curr_frame_id - self._last_frame_id
            self._stride_speed = self._count_windows * self._fps / self._frame_interval
            self._last_frame_id = self._curr_frame_id
            self._updata_speed = False

        return self._stride_speed
            
    @property
    def speed(self):
        return self._stride_speed



        
        
