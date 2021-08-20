from pymongo import MongoClient
import csv
from OpenPose import OpenPose
import cv2
from utils import crop_resize_image

db_user='user'
db_pass='PJdEDYM9Cx9xAfE0'
db_name='justdance'

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient("mongodb+srv://" + db_user + ":" + db_pass + "@cluster0.tjbly.mongodb.net/" + db_name + "?retryWrites=true&w=majority")
db=client.justdance

songs = db.songs

op = OpenPose()

cap = cv2.VideoCapture('video.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

joint_time = []


while(cap.isOpened()):
    frame_exists, curr_frame = cap.read()
    if frame_exists:
        data = op.process_image_keypoints(curr_frame)
        data = data[0, :15]
        print(data.shape)
        data = crop_resize_image(data)
        data = data.tolist()

        joint_time.append({"timestamp": cap.get(cv2.CAP_PROP_POS_MSEC), "joint_map": data })

    else:
        break

cap.release()

# print(joint_time[0])

x = songs.insert_one({"_id": 2, "name": "Dynamite", "artist":"BTS", "length": "0:50", "difficulty": "HARD", "data": joint_time})
# print(x.inserted_id)


# with open("out.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(joint_time)
