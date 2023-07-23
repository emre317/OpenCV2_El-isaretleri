
import cv2
import numpy as np
import os

Cam = cv2.VideoCapture(0)
kernel = np.ones((12,12),np.uint8)

# to compare the difference between the pictures
def PictureFind(Picture1,Picture2):
    Picture2 = cv2.resize(Picture2,(Picture1.shape[1],Picture1.shape[0]))
    Dif_Pic = cv2.absdiff(Picture1, Picture2)
    Dif_Number = cv2.countNonZero(Dif_Pic)
    return Dif_Number
    
# to load data and export to file   
def UploadData():
    Data_Names = []
    Data_Pictures = []
    
    Files = os.listdir("Data/")
    for File in Files:
        # After running the code to see only the name of the data in the console not to see the extension
        Data_Names.append(File.replace(".jpg","")) 
        Data_Pictures.append(cv2.imread("Data/"+File,0))
     
    return Data_Names,Data_Pictures

# to find the minimum values
def Classify(Pic,Data_Names,Data_Pictures):
    Min_Index = 0
    Min_Value = PictureFind(Pic, Data_Pictures[0])
    for t in range(len(Data_Names)):
        Dif_Value = PictureFind(Pic, Data_Pictures[t])
        if(Dif_Value<Min_Value):
            Min_Value=Dif_Value
            Min_Index=t
    return Data_Names[Min_Index]


Data_Names, Data_Pictures = UploadData()


while True:
    ret, Square = Cam.read()
    Cut_Square = Square[0:200,0:250]
    Cut_Square_Gray = cv2.cvtColor(Cut_Square, cv2.COLOR_BGR2GRAY)
    Cut_Square_HSV = cv2.cvtColor(Cut_Square, cv2.COLOR_BGR2HSV)
    
    Lower_Values = np.array([0,20,40])
    Upper_Values = np.array([40,255,255])
    
    Color_Filter_Result = cv2.inRange(Cut_Square_HSV, Lower_Values, Upper_Values)
    Color_Filter_Result = cv2.morphologyEx(Color_Filter_Result, cv2.MORPH_CLOSE, kernel)
    Color_Filter_Result = cv2.dilate(Color_Filter_Result, kernel, iterations=1)
    
    Result = Cut_Square.copy()
    
    cnts,_ = cv2.findContours(Color_Filter_Result,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    Max_Width = 0
    Max_Len = 0
    Max_Index = -1
    for t in range(len(cnts)):
        cnt = cnts[t]
        x,y,w,h = cv2.boundingRect(cnt)
        if (w>Max_Width and h>Max_Len):
            Max_Len = h
            Max_Width = w
            Max_Index = t
    
    if(len(cnts)>0):
        x,y,w,h = cv2.boundingRect(cnts[Max_Index])
        cv2.rectangle(Result,(x,y),(x+w,y+h),(0,255,0),2)
        Hand_Pic = Color_Filter_Result[y:y+h,x:x+w]
        cv2.imshow("Hand Picture", Hand_Pic)
        print(Classify(Hand_Pic, Data_Names, Data_Pictures))
        
    cv2.imshow("Square",Square)
    cv2.imshow("Cut Square", Cut_Square)
    cv2.imshow("Color Filter Result",Color_Filter_Result)
    cv2.imshow("Result",Result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
Cam.release()
cv2.destroyAllWindows() 

