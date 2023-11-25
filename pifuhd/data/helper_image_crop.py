import cv2
import numpy as np


def crop_image(img, rect):
    x, y, w, h = rect

    left = abs(x) if x < 0 else 0
    top = abs(y) if y < 0 else 0
    right = abs(img.shape[1] - (x + w)) if x + w >= img.shape[1] else 0
    bottom = abs(img.shape[0] - (y + h)) if y + h >= img.shape[0] else 0

    if img.shape[2] == 4:
        color = [0, 0, 0, 0]
    else:
        color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    x = x + left
    y = y + top

    return new_img[y:(y + h), x:(x + w), :]


def face_crop(pts):
    flag = pts[:, 2] > 0.2

    mshoulder = pts[1, :2]
    rear = pts[18, :2]
    lear = pts[17, :2]
    nose = pts[0, :2]

    center = np.copy(mshoulder)
    center[1] = min(nose[1] if flag[0] else 1e8, lear[1] if flag[17] else 1e8, rear[1] if flag[18] else 1e8)

    ps = []
    pts_id = [0, 15, 16, 17, 18]
    cnt = 0
    for i in pts_id:
        if flag[i]:
            ps.append(pts[i, :2])
            if i in [17, 18]:
                cnt += 1

    ps = np.stack(ps, 0)
    if ps.shape[0] <= 1:
        raise IOError('key points are not properly set')
    if ps.shape[0] <= 3 and cnt != 2:
        center = ps[-1]
    else:
        center = ps.mean(0)
    radius = int(1.4 * np.max(np.sqrt(((ps - center[None, :]) ** 2).reshape(-1, 2).sum(0))))

    # radius = np.max(np.sqrt(((center[None] - np.stack([]))**2).sum(0))
    # radius = int(1.0*abs(center[1] - mshoulder[1]))
    center = center.astype(int)

    x1 = center[0] - radius
    x2 = center[0] + radius
    y1 = center[1] - radius
    y2 = center[1] + radius

    return x1, y1, x2 - x1, y2 - y1


def upperbody_crop(pts):
    flag = pts[:, 2] > 0.2

    mshoulder = pts[1, :2]
    ps = []
    pts_id = [8]
    for i in pts_id:
        if flag[i]:
            ps.append(pts[i, :2])

    center = mshoulder
    if len(ps) == 1:
        ps = np.stack(ps, 0)
        radius = int(0.8 * np.max(np.sqrt(((ps - center[None, :]) ** 2).reshape(-1, 2).sum(1))))
    else:
        ps = []
        pts_id = [0, 2, 5]
        ratio = [0.4, 0.3, 0.3]
        for i in pts_id:
            if flag[i]:
                ps.append(pts[i, :2])
        ps = np.stack(ps, 0)
        radius = int(0.8 * np.max(np.sqrt(((ps - center[None, :]) ** 2).reshape(-1, 2).sum(1)) / np.array(ratio)))

    center = center.astype(int)

    x1 = center[0] - radius
    x2 = center[0] + radius
    y1 = center[1] - radius
    y2 = center[1] + radius

    return x1, y1, x2 - x1, y2 - y1


def fullbody_crop(pts):
    flags = pts[:, 2] > 0.5  # openpose
    # flags = pts[:,2] > 0.2  #detectron
    check_id = [11, 19, 21, 22]
    cnt = sum(flags[check_id])

    if cnt == 0:
        center = pts[8, :2].astype(int)
        pts = pts[pts[:, 2] > 0.5][:, :2]
        radius = int(1.45 * np.sqrt(((center[None, :] - pts) ** 2).sum(1)).max(0))
        center[1] += int(0.05 * radius)
    else:
        pts = pts[pts[:, 2] > 0.2]
        pmax = pts.max(0)
        pmin = pts.min(0)

        center = (0.5 * (pmax[:2] + pmin[:2])).astype(int)
        radius = int(0.65 * max(pmax[0] - pmin[0], pmax[1] - pmin[1]))

    x1 = center[0] - radius
    x2 = center[0] + radius
    y1 = center[1] - radius
    y2 = center[1] + radius

    return x1, y1, x2 - x1, y2 - y1

