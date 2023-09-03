#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2, time, os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
import matplotlib.pyplot as plt
from cvlib.object_detection import draw_bbox

import cvlib as cv

np.random.seed(20)

class Detector:
    def __init__(self):
        pass
    
    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()
            self.colorList = np.random.uniform(low = 0,high =255, size = (len(self.classesList),3))
            print(len(self.classesList),len(self.colorList))

    def downloadModel(self, modelURL):
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]
        print(fileName)
        print(self.modelName)
        self.cacheDir = "./pretrained_models"    
        os.makedirs(self.cacheDir, exist_ok = True)
        get_file(fname = fileName, origin=modelURL,cache_dir=self.cacheDir,cache_subdir="checkpoints",extract = True)
        
    def loadModel(self):
        print("Loading Model" + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir,"checkpoints",self.modelName,"saved_model"))
        print("\n")
        print("Model " + self.modelName + " loaded successfully!!")
        
    def createBoundingbox(self,image, threshold = 0.5):
        inputTensor = cv2.cvtColor(image.copy(),cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor,dtype = tf.uint8)
        inputTensor = inputTensor[tf.newaxis,...]
        detection = self.model(inputTensor)
        bboxs = detection['detection_boxes'][0].numpy()
        classIndexes = detection['detection_classes'][0].numpy().astype(np.int32)
        classScores = detection['detection_scores'][0].numpy()
        imH, imW, imC = image.shape
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size = 50,iou_threshold=threshold,score_threshold=threshold)
        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i])
                classConfidence = np.round(100*classScores[i])
                classIndex = classIndexes[i]
                classLabelText = self.classesList[classIndex].upper()
                classColor = self.colorList[classIndex]
                displayText = '{}: {}%'.format(classLabelText,classConfidence)
                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = (xmin*imW , xmax*imW, ymin*imH, ymax*imH)
                xmin, xmax, ymin, ymax = int(xmin),int(xmax), int(ymin),int(ymax)
                cv2.rectangle(image, (xmin,ymin),(xmax,ymax),color = classColor,thickness = 1)
                cv2.putText(image, displayText,(xmin , ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor,2)
                lineWidth = min(int((xmax - xmin)*0.2), int((ymax - ymin)*0.2))
                cv2.line(image,(xmin,ymin),(xmin+lineWidth,ymin),classColor,thickness = 5)
                cv2.line(image,(xmin,ymin),(xmin,ymin+lineWidth),classColor,thickness = 5)
                cv2.line(image,(xmax,ymin),(xmax-lineWidth,ymin),classColor,thickness = 5)
                cv2.line(image,(xmax,ymin),(xmax,ymin+lineWidth),classColor,thickness = 5)
                #########
                cv2.line(image,(xmin,ymax),(xmin+lineWidth,ymax),classColor,thickness = 5)
                cv2.line(image,(xmin,ymax),(xmin,ymax-lineWidth),classColor,thickness = 5)
                cv2.line(image,(xmax,ymax),(xmax-lineWidth,ymax),classColor,thickness = 5)
                cv2.line(image,(xmax,ymax),(xmax,ymin-lineWidth),classColor,thickness = 5)
        return image      
                
        
        
    def predictImage(self, imagePath,threshold = 0.5):
        image = cv2.imread(imagePath)
        bboxImage = self.createBoundingbox(image,threshold)
        cv2.imwrite(self.modelName + ".jpeg",bboxImage)
        cv2.imshow("Result",bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def predictVideo(self,videoPath, threshold = 0.5):
        cap = cv2.VideoCapture(videoPath)
        if(cap.isOpened() == False):
            print("Error opening file .......")
            return
        (success, image) = cap.read()
        startTime = 0
        while success:
            labels = []
            currentTime = time.time()
            fps = 1/(currentTime - startTime)
            startTime = currentTime
            ret, frame = cap.read()

            # Detect objects and draw on screen
            bbox, label, conf = cv.detect_common_objects(frame)
            
            bboxImage = self.createBoundingbox(frame,threshold)
            cv2.putText(bboxImage,"FPS : "+str(int(fps)),(20,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
            cv2.imshow("Result",bboxImage)
            for item in label:
                if item in labels:
                    break
                else:
                    labels.append(item)
            print(labels)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success,image) = cap.read()
        cv2.destroyAllWindows()
        
   


# In[3]:


modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz"

classFile = "C:/Users/perep/OneDrive/Desktop/spring 2023/programming lamguages/project/project2/coco.names"
threshold = 0.5


# 

# In[4]:


imagePth = "C:/Users/perep/OneDrive/Desktop/spring 2023/programming lamguages/project/project2/"


# In[5]:


videoPth ="C:/Users/perep/OneDrive/Desktop/spring 2023/programming lamguages/project/project2/" 


# In[6]:


detect = Detector()
detect.readClasses(classFile)
detect.downloadModel(modelURL)
detect.loadModel()

#detect.predictVideo(videoPath,threshold)


# In[9]:


choice = "y"
while (choice == "y"):
    print("1.enter your selection for object detection")
    print("1. Image\n2. Video\n 3. Camera\n")
    n = int(input("enter your choice: "))
    if(n == 1):
        path = input("enter your image name with extension : ")
        ImagePath = imagePth + path
        print(ImagePath)
        detect.predictImage(ImagePath,threshold)
    elif(n == 2):
        path = input("enter your video name with extension : ")
        VideoPath = videoPth + path
        detect.predictVideo(VideoPath,threshold)
    elif(n == 3):
        detect.predictVideo(0,threshold)
    else:
        print("Sorry,you have entered the wrong number!")
    choice = input("If you want to continue please enter y, else please enter n")
print("Thank you !!!")   


# In[ ]:




