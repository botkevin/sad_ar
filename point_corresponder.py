from skvideo.io import vread, vwrite
import numpy as np
from matplotlib import pyplot as plt
import cv2
import csv
from tqdm import tqdm

def track_points (vid, points):
    trackers = []
    for i in range(len(points)):
        trackers.append(cv2.TrackerMedianFlow_create())

    frame = vid[0]

    new_vid=[vid[0]]

    # initial bounding box
    box_len = 20
    bboxes = []
    for point in points:
        bbox = point - [box_len//2,box_len//2]
        bbox = np.concatenate((bbox, [box_len,box_len]))
        bbox = tuple(bbox.astype(int))
        bboxes.append(bbox)

    # Initialize tracker with first frame and bounding box
    for i in range(len(points)):
        bbox = bboxes[i]
        tracker = trackers[i]
        ok = tracker.init(frame, bbox)
        if not ok:
            raise Exception

    # init empty
    tracked_points = np.zeros((len(vid),len(points),2))
    tracked_points[0] = points

    for fi, frame in tqdm(enumerate(vid[1:])):
        
        frame = frame.copy()

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        oks = [True]*len(points)
        for i in range(len(points)):
            tracker = trackers[i]
            ok, bbox = tracker.update(frame)
            oks[i] = ok
            bboxes[i] = bbox

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        for i in range(len(points)):
            ok = oks[i]
            if ok: 
                # Tracking success
                bbox = bboxes[i]
                tracked_points[fi,i,:] = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Display result
        new_vid.append(frame)

    return new_vid, tracked_points

def findPerspective(p1, p2):
    # src p1 3, dst p2 2
    A = []
    b = []
    for i in range(len(p1)):
        if not (p2[i] == [0,0]).all():
            # good point
            x, y, z = p1[i]
            u, v = p2[i] 
            ai = [
                    [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z],
                    [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z]
                ]
            bi = [[u],
                [v]]
            A += ai
            b += bi
    A = np.asarray(A)
    b = np.asarray(b)
    Hv = np.vstack((np.linalg.lstsq(A, b, rcond=None)[0], np.asarray([1])))
    H = Hv.reshape((3,4))
    return H    

def warp_points (points, H):
    p = np.vstack((points.T, np.ones((1,points.shape[0]))))
    p1 = (H @ p)
    p1 = p1/p1[2,:]
    
    return p1.T[:,:2]

def draw_axis (img, H):
    p_axis = np.asarray([[0,0,0], [10,0,0], [0,10,0], [0,0,10]])
    H_p_axis = warp_points (p_axis, H).astype(int)
    corner = tuple(H_p_axis[0].ravel())
    img = cv2.line(img, corner, tuple(H_p_axis[1].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(H_p_axis[2].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(H_p_axis[3].ravel()), (0,0,255), 5)
    return img

def draw_cube(img, H):
    axis = np.asarray([[0,0,0], [0,4,0], [4,4,0], [4,0,0],
                   [0,0,4],[0,4,4],[4,4,4],[4,0,4] ]) + 4
    imgpts = warp_points (axis, H)
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

def draw (vid, points, p_world, draw_fn):
    new_frames = []
    for fi in tqdm(range(len(vid))):
        frame = vid[fi].copy()
        point = points[fi]
        if point.any():
            H = findPerspective (p_world, point)
            new_frame = draw_fn (frame, H)
        else:
            new_frame = frame
        new_frames.append (new_frame)
    return new_frames