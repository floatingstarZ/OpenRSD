import cv2
import numpy as np
import cv2
import colorsys

def bbox2mid(bbox):
    """
    get middle point
    :param bbox: top left, right down
    :return: mid point
    """
    [x1, y1, x2, y2] = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_ar(bbox):
    """
    
    :param bbox: top left, right down
    :return: aspect ratio
    """
    [x1, y1, x2, y2] = bbox
    return (y2 - y1) / (x2 - x1)

def draw_mid(img, bbox, radio, color, thickness=2, text='', y_bias=5):
    """

    :param bbox:  top left, right down
    :return:
    """
    (xc, yc) = bbox2mid(bbox)
    cv2.circle(img, (xc, yc), radio, color=color, thickness=thickness)
    if text:
        cv2.putText(img, text,
                    (xc, yc - y_bias), cv2.FONT_HERSHEY_COMPLEX, 0.3, color,
                    thickness=1)

def draw_matched_line(img, gt, dt, line_color):
    """

    :param gt: left top, right down
    :param dt: same as gt
    :param line_color:
    :return:
    """
    # [xl_g, yl_g, xr_g, yr_g] = gt
    # [xl_d, yl_d, xr_d, yr_d] = dt
    def form_corner(x0, y0, x1, y1):
        return np.array([(x0, y0), (x0, y1), (x1, y0), (x1, y1)])
    gt_corners = form_corner(*gt)
    dt_corners = form_corner(*dt)
    for i, gt_corner in enumerate(gt_corners):
        cv2.line(img, tuple(gt_corner), tuple(dt_corners[i]),
                 color=line_color, thickness=1)

def draw_bbox_ltrd(img, bbox, rect_color, thickness=1, text='', font_size=0.5):
    (x1, y1, x2, y2) = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), rect_color, thickness=thickness)
    if text:
        cv2.putText(img, text,
                    (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, font_size, rect_color, 2)


def draw_bbox(img, bbox, rect_color):
    (x1, y1, x2, y2) = bbox
    (x1, y1) = (int(x1), int(y1))
    (x2, y2) = (int(x2), int(y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), rect_color, thickness=1)


def draw_gt(img, bbox, cat, id_name_map, rect_color, text_color=(0, 200, 200),
            rect_thick=2, font_size=0.5):
    (x1, y1, x2, y2) = bbox
    (x1, y1) = (int(x1), int(y1))
    (x2, y2) = (int(x2), int(y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), rect_color, thickness=rect_thick)
    cv2.putText(img, id_name_map[str(cat)],
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, 2)


def draw_dt(img, bbox, cat, score, id_name_map, rect_color, text_color=(0, 200, 200),
            no_text=False):
    (x1, y1, x2, y2) = bbox
    (x1, y1) = (int(x1), int(y1))
    (x2, y2) = (int(x2), int(y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), rect_color, thickness=1)
    show_text = '%s%.3f' % (id_name_map[str(cat)], score)
    if not no_text:
        cv2.putText(img, show_text,
                    (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.3, text_color,
                    thickness=1)

def draw_bboxes(img, bbs, color, thickness=1, texts=None, font_size=0.5):
    """

    :param img: 
    :param bbs: tensor
    :param color: 
    :param thickness: 
    :param texts: 
    :param font_size: 
    :return: 
    """
    if texts:
        for b, t in zip(bbs, texts):
            b = b.numpy()
            draw_bbox_ltrd(img, b, color, text=t, thickness=thickness, font_size=font_size)
    else:
        for b in bbs:
            b = b.numpy()
            draw_bbox_ltrd(img, b, color, thickness=thickness, font_size=font_size)


# class BboxDrawer:
#     def __init__(self, type='val2017'):
#         from configs.config import bbox_dict_file_path
#         from utils.sl_utils import jsonload
#         self.bbox_dict = jsonload(bbox_dict_file_path)
#         from configs.id2name import id2name
#         self.id2name = id2name
#         from configs.color_map import gt_colors
#         self.colors = gt_colors['matched']
#
#
#     def draw(self, img_id, org_img=np.empty([0]), scale=(1, 1)):
#         try:
#             info = self.bbox_dict[str(img_id)]
#             file_path = info['file_path']
#             bboxes = info['bbox']
#             if not org_img.any():
#                 org_img = cv2.imread(file_path)
#             bboxes = np.array(bboxes)
#             bboxes[:, 0] = scale[0] * bboxes[:, 0]
#             bboxes[:, 2] = scale[0] * bboxes[:, 2]
#             bboxes[:, 1] = scale[1] * bboxes[:, 1]
#             bboxes[:, 3] = scale[1] * bboxes[:, 3]
#         except IndexError:
#             print(self.bbox_dict[str(img_id)])
#             return org_img
#
#         for i, bbox in enumerate(bboxes):
#             draw_gt(org_img, bbox[0: 4], int(bbox[4]), self.id2name, (255, 200, 0),
#                     text_color=(0, 200, 200), rect_thick=1, font_size=0.5)
#         return org_img
#
#
#     def draw_SML(self, img_id, org_img=np.empty([0]), scale=(1, 1)):
#         """
#         different area with different color
#         :param img_id:
#         :param org_img:
#         :param scale:
#         :return:
#         """
#         try:
#             info = self.bbox_dict[str(img_id)]
#             file_path = info['file_path']
#             org_bboxes = info['bbox']
#             if not org_img.any():
#                 org_img = cv2.imread(file_path)
#             bboxes = np.array(org_bboxes)
#             bboxes[:, 0] = scale[0] * bboxes[:, 0]
#             bboxes[:, 2] = scale[0] * bboxes[:, 2]
#             bboxes[:, 1] = scale[1] * bboxes[:, 1]
#             bboxes[:, 3] = scale[1] * bboxes[:, 3]
#         except IndexError:
#             print(self.bbox_dict[str(img_id)])
#             return org_img
#
#         for i, bbox in enumerate(bboxes):
#             [x, y, w, h] = org_bboxes[i][0: 4]
#             area = w * h
#             ids, label = get_rng(area)
#             color = self.colors[label]
#             draw_gt(org_img, bbox[0: 4], int(bbox[4]), self.id2name, color,
#                     text_color=color, rect_thick=1, font_size=0.5)
#         return org_img

if __name__ == '__main__':
    pass






