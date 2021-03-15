#<-------- Imports -------->#
import cv2
import numpy as np

#<-------- Computer Vision -------->#
class CV:
    def __init__(self, vid_path):
        self.video_feed = cv2.VideoCapture(vid_path)

    # Resize the frame
    @staticmethod
    def resize_frame(img, scale_percent=50):
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    # Find ball in frame
    def find_ball(self, frame):
        # Convert to HSV color scheme
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold HSV image to find red points
        H_range = (160, 20)
        S_range = (100, 220)
        V_range = (100, 220)
        
        below_180 = cv2.inRange(frame_HSV, (H_range[0], S_range[0], V_range[0]),
                                           (       255, S_range[1], V_range[1]))
        
        above_0 = cv2.inRange(frame_HSV, (         0, S_range[0], V_range[0]),
                                         (H_range[1], S_range[1], V_range[1]))

        frame_threshold = cv2.bitwise_or(below_180, above_0)

        # Find all non-black pixels 
        y_pts, x_pts = np.nonzero(frame_threshold)

        # Get indeces for the max and min y-values for pixels that aren't black
        max_y_ind = np.argmax(y_pts)
        min_y_ind = np.argmin(y_pts)

        # Find every point such that its y-value is equal to the max/min y-val then calculate mean of these x data points
        top_circle_line = np.argwhere(y_pts == y_pts[min_y_ind])
        bot_circle_line = np.argwhere(y_pts == y_pts[max_y_ind])

        mean_top_circle = int(round(np.mean(x_pts[top_circle_line])))
        mean_bot_circle = int(round(np.mean(x_pts[bot_circle_line])))

        self.bottom_ball.append((mean_bot_circle, y_pts[max_y_ind]))
        self.top_ball.append((mean_top_circle, y_pts[min_y_ind]))

    # Complete cv run
    def run_cv(self):
        final_frame = None
        self.bottom_ball = []
        self.top_ball =[]

        # Search each frame
        while(self.video_feed.isOpened()):
            # Get frame
            ret, frame = self.video_feed.read()

            if not ret:
                break

            # Find ball in frame
            self.find_ball(frame)

            # Draw frame with ball tops and bottoms
            for bot_ball_i, top_ball_i in zip(self.bottom_ball, self.top_ball):
                cv2.circle(frame, bot_ball_i, 6, 150, thickness=-1)
                cv2.circle(frame, top_ball_i, 6, 150, thickness=-1)

            # Display frame
            final_frame = frame
            frame = CV.resize_frame(frame, 40)
            cv2.imshow('editted frame', frame)

            k = cv2.waitKey(1) & 0xff
            if k == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return final_frame