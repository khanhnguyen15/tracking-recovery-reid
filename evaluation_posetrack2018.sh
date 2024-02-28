# Extracting the bounding boxes gallery
python -m track_recover.scripts.extract_bbox \
--input posetrack2018_openpifpaf/ \
--dataset 
--output bbox_gallery/


# Building the storage for all posetrack2018 validation sequence
python -m track_recover.scripts.get_storage \
--gallery bbox_gallery/ \
--storage storage/ \
--reid-model reid_model/best_model.pth.tar \


# Modifying the predictions
python -m track_recover.scripts.modify_predictions \
--input posetrack2018_openpifpaf/ \
--storage storage/ \
--output posetrack2018_modified/ \
--threshold 70.0 \
--max-rank 10 \
--memory-len 5 \
--mode recent


# Evaluating the modified predictions
python -m track_recover.scripts.evaluate_predictions \
--ground-truth data/data-posetrack2018/annotations/val/ \
--openpifpaf-pred posetrack2018_openpifpaf/ \
--modified-preds posetrack2018_modified/ \
--output evaluation/ \