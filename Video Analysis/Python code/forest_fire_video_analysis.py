#import opencv library
import cv2
#import numpy
import numpy as np
#import image function from keras
from tensorflow.keras.preprocessing import image
#import load_model from keras
from keras.models import load_model
#import Client from twilio API
from twilio.rest import Client
#import playsound package
from playsound import playsound
#from PIL import image
#load the saved model
model=load_model('../forest1.h5')
#define video
video=cv2.VideoCapture(0)
#define the features
name=['forest','with fire']
while(1):
  success,frame=video.read()
  cv2.imwrite("../ffproj/image.jpg",frame)
  img=image.load_img("../ffproj/image.jpg",target_size=(64,64))
  x=image.img_to_array(img)
  res = cv2.resize(x, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
  x=np.expand_dims(res,axis=0)
  pred=model.predict(x)
  p=pred[0]
  print(pred)
  cv2.putText(frame,"predicted class = "+str(p),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0))
  if pred[0]==1:
    #twilio account ssid
    account_sid='AC805dad0282a5eaebfa3149365f1c605d'
    #twilio account authentication token
    auth_token ='a27b535245ca70f9c7b4875865a620e1'
    client=Client(account_sid,auth_token)

    message=client.messages \
    .create(
        body='Forest Fire is detected,stay alert',
        #use twilio free number
        from_='+1 223 217 4143',
        #to number
        to='+91 6300780739')
    print(message.sid)
    print('Fire Detected')
    print('SMS sent!')
    playsound('../tornado-siren-in-streamwood-il-35510.mp3')
  else:
    print('No Danger')
    #break
  cv2.imshow("image",frame)
  if cv2.waitKey(1) & 0xFF == ord('a'):
     break
video.release()
cv2.destroyAllWindows()