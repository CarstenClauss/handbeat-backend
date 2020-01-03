import cv2 as cv
import numpy as np

from hand_beat import CascadeClassifierHandDetector


def main():
    cap = cv.VideoCapture(cv.CAP_V4L)

    cap.set(cv.CAP_PROP_GAIN, 0)
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, False)
    cap.set(cv.CAP_PROP_AUTO_WB, False)

    hand_detector = CascadeClassifierHandDetector()
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            hand_left, hand_right = hand_detector.detect(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))

            copy = np.copy(frame)

            if hand_left is not None:
                cv.drawMarker(copy, tuple(hand_left.astype(np.int)), (0, 255, 0), thickness=3)

            if hand_right is not None:
                cv.drawMarker(copy, tuple(hand_right.astype(np.int)), (255, 0, 0), thickness=3)

            cv.imshow('Hands', copy[:, ::-1])

        if cv.waitKey(10) == ord('q'):
            break

    cv.destroyAllWindows()

    cap.release()


if __name__ == '__main__':
    main()
