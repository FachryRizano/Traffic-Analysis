from tracker import Tracker
import imutils
import cv2
from google.colab import cv2_imshow

class Counter():
    def __init__(self):
        cap, fps, h, w = self.__init_capture()
        self.vid_writer = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
        self.left_id_arr = []
        self.right_id_arr = []
        
        # Truck, Bus, Car
        self.object_count ={'Truck':0,'Bus':0,'Car':0}
        
        self.tracker = Tracker(model='yolox-s', ckpt='/content/best_ckpt.pth.tar',filter_class=['truck','bus','car'])
        self.left_right_count = [0,0]

        self.left_border_origin= (0,int(2/3*h))
        self.left_border_end = (int(w/2)-50,int(2/3*h)+5)
        self.right_border_origin= (int(w/2)-25,int(2/3*h))
        self.right_border_end = (w-10,int(2/3*h)+5)
        self.start_count(cap,h,w)


    def __init_capture(self):
        cap = cv2.VideoCapture('/content/malam_sibuk_20.mp4')  # open one video
        fps = cap.get(cv2.CAP_PROP_FPS)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return cap, fps, h, w
    
    def start_count(self,cap,h,w):
        org = (50,50)
        for i in range(50):
            _, img = cap.read() # read frame from video
            if img is None:
                break
            # im = imutils.resize(img, height=500)
            image,outputs= self.tracker.update(img)
            #Left
            cv2.line(image,(0,int(2/3*h)),(int(w/2)-50,int(2/3*h)),(255,0,0),3)
            
            #Right
            cv2.line(image,(int(w/2),int(2/3*h)),(w-10,int(2/3*h)),(255,0,0),3)
            # print(outputs)
            for out in outputs:
                center_x= out[0] + (out[2] - out[0])/2
                center_y= out[1] + (out[3] - out[1])/2
                if out[-2] not in self.left_id_arr and self.left_border_origin[0]<=center_x<=self.left_border_end[1] and self.left_border_origin[1]<=center_y<=self.left_border_end[1]:
                    self.left_id_arr.append(out[-2])
                    if int(out[-1]) == 0:
                        self.object_count['Truck']+=1
                    elif int(out[-1]) == 1:
                        self.object_count['Bus']+=1        
                    elif int(out[-1]) == 2:    
                        self.object_count['Car']+=1
                    self.left_right_count[0]+=1
                if out[-2] not in self.right_id_arr and self.right_border_origin[0]<=center_x<=self.right_border_end[0] and self.right_border_origin[1]<=center_y<=self.right_border_end[1]:
                    self.right_id_arr.append(out[-2])
                    if int(out[-1]) == 0:
                        self.object_count['Truck']+=1
                    elif int(out[-1]) == 1:
                        self.object_count['Bus']+=1        
                    elif int(out[-1]) == 2:    
                        self.object_count['Car']+=1
                    self.left_right_count[1]+=1
                for key,value in self.object_count:
                  cv2.putText(image,f'{key}:{value}',org, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                  org[0] += 20
                cv2_imshow(image)
                im = imutils.resize(image, height=h,width=w)
                image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                self.vid_writer.write(image)
        cap.release()

if __name__ == "__main__":
    counter = Counter()
    print('Object Count:',counter.object_count)
    print('Left Right Count:', counter.left_right_count)

