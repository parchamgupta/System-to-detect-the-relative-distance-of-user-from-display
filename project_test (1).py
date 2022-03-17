from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
import pandas as pd
from tkinter import *
from tkinter import messagebox


def face_crop(img):
    detector = MTCNN()
    detail = detector.detect_faces(img)
    if not detail:
        resized_image = cv2.resize(img, (250, 250), interpolation=cv2.INTER_AREA)
        return resized_image, None
    dim = np.min([img.shape[0], img.shape[1]])

    x = int(detail[0]["box"][0])
    y = int(detail[0]["box"][1])
    w = int(detail[0]["box"][2])
    h = int(detail[0]["box"][3])
    center_x = x + w // 2
    center_y = y + h // 2

    left = center_x - dim // 2
    if left < 0:
        left = 0
    right = left + dim
    if right > img.shape[1]:
        right = img.shape[1]
        left = right - dim

    top = center_y - dim // 2
    if top < 0:
        top = 0
    bottom = top + dim
    if bottom > img.shape[0]:
        bottom = img.shape[0]
        top = bottom - dim

    crop_image = img[int(top):int(bottom), int(left):int(right)]
    resized_image = cv2.resize(crop_image, (250, 250), interpolation=cv2.INTER_AREA)
    detail = detector.detect_faces(resized_image)
    return resized_image, detail


def draw_facebox(filename, result_list):
    for result in result_list:
        x, y, w, h = result['box']
        cv2.rectangle(filename, (x, y), (x + w, y + h), (255, 0, 0), 2)


def configure(bt):
    cap = cv2.VideoCapture(0)
    img1 = cv2.imread("C:\\Users\\jainn\\OneDrive\\Desktop\\img2.jpg")
    img1 = cv2.resize(img1, (250, 250))
    width = []
    height = []

    messagebox.showinfo('Configure', 'Please sit at an appropriate viewing distance (2 ft.) from the screen and then '
                                     'click OK')

    for i in range(5):
        ret, frame = cap.read()
        if ret:
            image, f_detail = face_crop(frame)
            if f_detail is None:
                continue
            else:
                width.append(f_detail[0]['box'][2])
                height.append(f_detail[0]['box'][3])

            dst = cv2.addWeighted(image, 1, img1, 0.2, 0)
            cv2.imshow("My frame", dst)

        key = cv2.waitKey(500)
        if key == ord("q"):
            break

    width = np.asarray(width)
    height = np.asarray(height)
    mean_w = width.mean()
    mean_h = height.mean()

    df = pd.read_csv("C:/Users/jainn/Downloads/Mean_values.csv")

    global h_factor
    global w_factor
    global eyes
    global mouth

    h_factor = df['height'][0] / mean_h
    w_factor = df['width'][0] / mean_w
    eyes = df['eyes'][0]
    mouth = df['mouth_eyes'][0]

    # cap.release()
    cv2.destroyAllWindows()

    messagebox.showinfo('Configured', 'CLick RUN to start the program.')
    bt.configure(state=NORMAL)


def run(window, h_fact, w_fact, eye, m):
    window.destroy()
    count = 0
    pos_count = 0
    while True:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            image, f_detail = face_crop(frame)

            if f_detail is None:
                continue
            else:
                draw_facebox(image, f_detail)
                lex = f_detail[0]["keypoints"]["left_eye"][0]
                ley = f_detail[0]["keypoints"]["left_eye"][1]
                rex = f_detail[0]["keypoints"]["right_eye"][0]
                rey = f_detail[0]["keypoints"]["right_eye"][1]
                mlx = f_detail[0]["keypoints"]["mouth_left"][0]
                mly = f_detail[0]["keypoints"]["mouth_left"][1]
                mrx = f_detail[0]["keypoints"]["mouth_right"][0]
                mry = f_detail[0]["keypoints"]["mouth_right"][1]

                btw_eyes = np.sqrt((rex - lex) ** 2 + (rey - ley) ** 2) * w_fact
                mouth_eye = ((mry + mly) / 2 - (ley + rey) / 2) * h_fact

                if 20.0 > btw_eyes - eye > -5.0 and 20.0 > mouth_eye - m > -5.0:
                    print("0")
                    count = 0
                elif btw_eyes - eye > 20.0 and mouth_eye - m > 20.0:
                    print("-1")
                    if count > 0:
                        count = 0
                    else:
                        count = count - 1
                elif btw_eyes - eye < -5.0 and mouth_eye - m < -5.0:
                    print("1")
                    if count < 0:
                        count = 0
                    else:
                        count = count + 1
                elif 2.0 > btw_eyes - eye > -2.0:
                    print("0")
                    count = 0
                else:
                    # count = 0
                    print("*")

                if count >= 6:
                    print("Come closer")
                    cv2.rectangle(image, (0, 0), (250, 250), (255, 0, 0), 20)
                    cv2.putText(image, 'Come Closer', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2,
                                cv2.LINE_AA)
                    # myCanvas.create_oval(10, 110, 90, 190, width=2, fill='blue')
                    # label1 = Label(window, bg='blue', relief=RAISED)
                    # label1.configure(height=5, width=11)
                    # label1.place(x=10, y=110)
                elif count <= -6:
                    print("Get farther")
                    cv2.rectangle(image, (0, 0), (250, 250), (0, 0, 255), 20)
                    cv2.putText(image, 'Get Farther', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2,
                                cv2.LINE_AA)
                    # label2 = Label(window, bg='red', relief=RAISED)
                    # label2.configure(height=5, width=11)
                    # label2.place(x=10, y=110)
                else:
                    print("Just Right")
                    cv2.rectangle(image, (0, 0), (250, 250), (0, 128, 0), 20)
                    cv2.putText(image, 'Just Right', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2,
                                cv2.LINE_AA)

                    # label3 = Label(window, bg='dark green', relief=RAISED)
                    # label3.configure(height=5, width=11)
                    # label3.place(x=10, y=110)

                if abs((mlx + mrx)//2 - (rex + lex)//2) >= 8 or abs((rey - ley)) >= 5:
                    print('#')
                    pos_count += 1
                else:
                    print('-')
                    pos_count = 0

                if pos_count >= 4:
                    print("Incorrect Posture")
                    cv2.putText(image, 'Incorrect Posture', (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2,
                                cv2.LINE_AA)

            cv2.imshow("My Screen", image)

        # key = cv2.waitKey(5000)
        key = cv2.waitKey(2000)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def gui():
    window1 = Tk()
    window1.geometry('320x240')
    window1.title('Indicator')
    # myCanvas = Canvas(window1, width=100, height=200)
    # myCanvas.grid()
    # lab = Label(window1, relief=RAISED)
    # lab.configure(height=5, width=11)
    # lab.place(x=10, y=110)
    #
    l1 = Label(window1, text='MOVE BACK', fg='white', bg='red', relief=RAISED)
    l1.configure(height=2, width=27)
    l1.place(x=65, y=130)
    l2 = Label(window1, text='JUST RIGHT', fg='white', bg='dark green', relief=RAISED)
    l2.configure(height=2, width=27)
    l2.place(x=65, y=165)
    l3 = Label(window1, text='COME CLOSE', fg='white', bg='blue', relief=RAISED)
    l3.configure(height=2, width=27)
    l3.place(x=65, y=200)
    l4 = Label(window1, text='Username : ')
    l4.place(x=72, y=10)
    l5 = Label(window1, text='Password : ')
    l5.place(x=72, y=35)
    txt1 = Entry(window1, width=18)
    txt1.place(x=140, y=10)
    txt2 = Entry(window1, width=18, show='*')
    txt2.place(x=140, y=35)
    bt2 = Button(window1, text='RUN', command=lambda: run(window1, h_factor, w_factor, eyes, mouth))
    bt2.config(height=1, width=10)
    bt2.place(x=122, y=95)
    bt2.configure(state=DISABLED)
    bt1 = Button(window1, text='CONFIGURE', command=lambda: configure(bt2))
    bt1.config(height=1, width=10)
    bt1.place(x=122, y=67)

    window1.mainloop()


if __name__ == '__main__':
    h_factor = 0
    w_factor = 0
    eyes = 0
    mouth = 0
    gui()
