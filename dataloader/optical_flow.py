import cv2
from PIL import Image


def extract_optical_flow(fname1, fname2):
    frame1 = cv2.imread(fname1)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.imread(fname2)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    u, v = flow[..., 0], flow[..., 1]
    u = Image.fromarray(u)
    v = Image.fromarray(v)
    return u, v


if __name__ == "__main__":
    fname1 = "d:/MyFile/dataset/UCF-101-splited/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/v_ApplyEyeMakeup_g01_c01_1.jpg"
    fname2 = "d:/MyFile/dataset/UCF-101-splited/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/v_ApplyEyeMakeup_g01_c01_2.jpg"
    H, V = extract_optical_flow(fname1, fname2)

    cv2.imshow("H", H)
    cv2.imshow("V", V)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
