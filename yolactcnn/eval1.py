from .data import COCODetection, get_label_map, MEANS, COLORS, cfg, set_cfg, set_dataset
from .yolact import Yolact
from .utils.augmentations import BaseTransform, FastBaseTransform, Resize
from .layers.box_utils import jaccard, center_size
from .utils import timer
from .utils.functions import SavePath
from .layers.output_utils import postprocess, undo_image_transformation

import time, random, json, os, cv2, warnings, requests, io 
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import defaultdict
from pathlib import Path
from collections import OrderedDict
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import imutils

from imutils import contours

from .eval_segm import mean_IU


warnings.filterwarnings("ignore")

color_cache = defaultdict(lambda: {})

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, top_k=10, score_threshold=0.003):

    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        t = postprocess(dets_out, w, h, visualize_lincomb = False,
                                        crop_masks        = True,
                                        score_threshold   = score_threshold)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
    classes, scores, boxes = [x[:top_k].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(top_k, classes.shape[0])


    for j in range(num_dets_to_consider):
        if scores[j] < score_threshold:
            num_dets_to_consider = j
            break 

    # Masks are drawn on the GPU, so don't copy
    masks = t[3][:top_k]
    image_mask = masks

    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return (img_gpu * 255).byte().cpu().numpy()

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = torch.Tensor(color).float() / 255.
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
    # After this, mask is of size [num_dets, h, w, 1]
    masks = masks[:num_dets_to_consider, :, :, None]
   
    # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
    if torch.cuda.is_available():
        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
    else:
        colors = torch.cat([get_color(j).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
    masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

    # This is 1 everywhere except for 1-mask_alpha where the mask is
    inv_alph_masks = masks * (-mask_alpha) + 1
    
    # I did the math for this on pen and paper. This whole block should be equivalent to:
    for j in range(num_dets_to_consider):
        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]

    # masks_color_summand = masks_color[0]
    # if num_dets_to_consider > 1:
    #     inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
    #     masks_color_cumul = masks_color[1:] * inv_alph_cumul
    #     masks_color_summand += masks_color_cumul.sum(dim=0)
    #img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        
    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()
    
    for j in reversed(range(num_dets_to_consider)):
        x1, y1, x2, y2 = boxes[j, :]
        color = tuple([int(x) for x in (get_color(j) * 255).numpy().astype(np.uint8)])
        score = scores[j]

        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

        _class = cfg.dataset.class_names[classes[j]]
        text_str = '%s: %.2f' % (_class, score) 

        #text_str = '%s: %.2f' % (_class, score) if args.display_scores else _class

        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1

        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

        text_pt = (x1, y1 - 3)
        text_color = [255, 255, 255]

        cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
        cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    print("Info: ", top_k, masks.size())
    print("Masks:", masks.size())

    return img_numpy, image_mask.cpu().numpy(), classes, scores, boxes


def bb_intersection_over_union(boxA, boxB):
    #https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def startinf(patinfo, imagedata, returnimg):

    with torch.no_grad():
        print('\n##############', '----  inferencing ----', '##############', flush=True)

        time1= time.time()
        frame = torch.from_numpy(imagedata)

        if torch.cuda.is_available():
            frame = frame.cuda()

        frame = frame.float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = net(batch)

        #return "OK"
        #print("Type: ", type(preds), preds, preds[0])

        if (preds[0] is not None):

            #print("Preds length: ", len(preds))
            #print(preds[0]['class'], preds[0]['box'], preds[0]['score'])
            #try:
            img_numpy, image_mask, classes, scores, boxes = prep_display(preds, frame, None, None, undo_transform=False, top_k= 10)

            l, h, w = image_mask.shape


            time2= time.time()
            print('Time taken for prediction the image: ', time2-time1, ' seconds', "Image mask shape: ", image_mask.shape, flush=True)


            print("returnimg: ", returnimg, "len boxes: ", len(boxes) )

            if (len(boxes) > 1):

                #box_mask1= np.zeros((h, w)).astype('uint8')
                #box_mask2= np.zeros((h, w)).astype('uint8')
                #cv2.rectangle(box_mask1, (boxes[0][0], boxes[0][1]), (boxes[0][2],boxes[0][3]), (255, 255,255), -1)
                #cv2.rectangle(box_mask2, (boxes[1][0], boxes[1][1]), (boxes[1][2],boxes[1][3]), (255, 255,255), -1)

                iou= bb_intersection_over_union(boxes[0], boxes[1])
                print("Bbox coords: ", boxes, "Intersection ove union: " ,iou)

                if (returnimg =="yes"):
                    return img_numpy

                if (iou > 0.4):
                    print("IoU suggests pathology....")
                    cv2.circle(img_numpy,(round((boxes[0][0] + boxes[1][0])/2), round((boxes[0][1] + boxes[1][1])/2)), 200, (255,120,0), 3)
                    

                    return "Anomalydetected"
                else:
                    return "OK"

            elif (len(boxes) == 1):

                if (returnimg =="yes"):
                    return img_numpy

                else:
                    return "OK"


        else:
            if (returnimg =="yes"):
                return imagedata

            else:
                return "NoResult"

        #savesrc = 'C:\\Users\\Sanyi\\Desktop\\WPy64-3860\\projects\\' + patinfo["InstanceUID"] + '.jpg'
        savesrc = 'C:\\Users\\Sanyi\\Desktop\\WPy64-3860\\projects\\testt.jpg'
        cv2.imwrite(savesrc, imagedata)


        #if image_mask.shape[0] > 3:
            #vert_data = start_post_processing(image_mask, jsonfilename)

    return "OK"

### -------------- start of file -----------------

with torch.no_grad():

    torch.set_default_tensor_type('torch.FloatTensor')

    dataset = None

    print('Loading model...', end='')

    cfg.num_classes = 3
    cfg.dataset.class_names = ('ETT', 'Carina')
    cfg.dataset.label_map =  {1:1, 2:2}


    net = Yolact()
    map_location = 'cpu'
    #print("Path:", os.path.dirname(os.path.realpath(__file__)))
    #print("Path cwd:", os.getcwd())
    net.load_weights("./yolactcnn/weights/yolact_base_1630_75000.pth", map_location=torch.device('cpu'))
    net.eval()
    print(' Done.')

    net.detect.use_fast_nms = True
    cfg.mask_proto_debug = False