from tidecv import TIDE,datasets
import pycocotools.coco

tide = TIDE()
bbox_results = datasets.COCOResult('./excluded/post/yolov_s_local128.pkl'.replace('.pkl','_imdb2coco.json')) # These files were downloaded above.
gt = datasets.COCO('./excluded/post/vid_gt_coco.json')
# bbox_results_ori = datasets.COCOResult('./ori_pred.json') # These files were downloaded above.
# gt_ori = datasets.COCO('./gt_ori.json')

tide.evaluate_range(gt, bbox_results, mode=TIDE.BOX) # Use TIDE.MASK for masks
tide.summarize()  # Summarize the results as tables in the console
tide.plot()