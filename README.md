## Online Identity Recovery with Deep Re-Identification

This repo contains the code for 2022 Spring semeter project on "Tracking Recovery with Re-Identification" at [EPFL VITA lab](https://www.epfl.ch/labs/vita/).

For more information, please refers to the project report and slides, which can be found in [docs](./docs). 

This project implemented a framework that aim to improve the tracking performances of [OpenPifPaf](https://vita-epfl.github.io/openpifpaf/dev/intro.html). It specifically addressed the problems ID assignments in OpenPifPaf and provide a method to to recovers the ID of an unique identity across frames. However, the framework has been extended to generally work with other detection systems.

The project is mainly tested on OpenPifPaf. Please follow the documentation [here](https://openpifpaf.github.io/intro.html) to learn about the package.


### Installation

Clone this repository in order to use it

```bash
# To clone the repository using HTTPS
git clone https://github.com/vita-epfl/tracking-recovery-reid.git
cd tracking-recovery-reid/
```

Install the dependency in the `requirements.txt` file

```bash
# To install dependencies
pip3 install -r requirements.txt
```

### Deep Re-Identification Model

The projects make use of a Deep Re-Identification model to recover the IDs. Follow this [link](https://github.com/vita-epfl/Deep-Visual-Re-Identification-with-Confidence) to retrain the model or alternative download it from [here](https://drive.google.com/file/d/1mk2C_vx6q-jC8upyB9XEiemJr2kFRxL5/view?usp=sharing). Upon downloading the pre-trained model, put it into a folder under root directory of your repo as following

```
root
  |___ docs 
  |___ track_recover 
  |___ reid_model 
    |___ best_model.pth.tar
  
```

### APIs

The `track_recover` package has two main components: Storage and IDTracker. To use the APIs, another component is necessary, which is the FeatureExtractor for the Deep Re-ID model.

Follow the following steps to call the APIs in an online manner:

First, initialize the components:
```python
# import the components
from . import track_recover

# set the model path and intialize the FeatureExtractor
model_path = 'reid_model/best_model.pth.tar'            # path/to/deep-re-id/model
extractor = track_recover.FeatureExtractor(model_path=model_path)

# initialize the storage and passing the extractor as the argument
storage = track_recover.Storage(extractor=extractor)

# initialize the IDTracker
id_tracker = track_recover.IDTracker()
```

Then, iteratively update the Storage and IDTracker for each frame predictions: 
```python
# the loops for predictions:
for image, predictions in frames:
    # update storage and extract features
    storage.update_frame(image, predictions)

    # get the features for every identities in the last frame 
    frame_id, input_IDs = storage.get_last_frame()

    #update the ID tracker and modify the predictions
    id_tracker.update(input_IDs=input_IDs, frame=frame_id)
    id_tracker.modify_annotations(annotations=predictions, frame=frame_id)
```

Note that the predictions must be list of dictionaries. Each dictionary is the detection and tracking of one identity and contains two keys: `bbox` which store the detection bbox of each identity in the format `(x, y, w, h)`; `id_` which store the track ID of the identity.

We provide `video.py` code to run the framework on OpenPifPaf. In order to run the code, go to [OpenPifPaf DEV repo](https://github.com/vita-epfl/openpifpaf) and clone it. Put the `track_recover` package inside the source code folder /src/openpifpaf/. Replace the `video.py` with the file within this repo. Run `pip install --editable '.[dev,train,test]'` to install the package from source. And run the standard openpifpaf.video command as documented [here](https://vita-epfl.github.io/openpifpaf/dev/predict_cli.html) with additional arguments:

```bash
python -m openpifpaf.video \
--checkpoint tshufflenetv2k30 \
--source myvideo.mp4 \
--video-output /path/to/the/video/output.mp4 \
--json-output /path/to/the/json/output.json \
--reid-model /path/to/the/deep-re-id/model.pth.tar \
--decoder=trackingpose:0
```

### Sample results

Here are the sample results from OpenPifPaf with the usage of the APIs

![openpifpaf](docs/openpifpaf_apis.gif)

Here are additional tracking recovery results from Posetrack2018 validation set. The sequence from the left is from original prediction and on the right is from the recovered version. The green color means the tracked ID matches the ground truth ID.

<p float="left">
  <img src="docs/001735_original.gif" width="300" />
  <img src="docs/001735_true.gif" width="300" /> 
</p>

<p float="left">
  <img src="docs/000532_original.gif" width="300" />
  <img src="docs/000532_true.gif" width="300" /> 
</p>

### Evaluation for Posetrack2018

To run evaluation on Posetrack2018 validation set with the best method, first download the Posetrack2018 dataset [here](https://posetrack.net/) and put in /data/data-posetrack2018/

Follow OpenPifPaf documentation to run benchmark on Posetrack to obtain the predictions for the validation set. Alternatively, download the predictions folder [here](https://drive.google.com/drive/folders/1Mt0dF2N397dBRnwlv3Q4gA2joUlzlyfc?usp=sharing). Put the predictions in the root directory.

To evaluate, run the script:
```bash
sh evaluate_posetrack2018.sh
```
The script consists of several python scripts needed to run the evaluation. Upon finishing, your directory would looks like this:

```
root
  |___ bbox_gallery             # gallery of bounding boxes for each identity for each sequence
  |___ data                     # create this folder to store your datasets, or create a symlink
    |___ data-posetrack2018
  |___ docs                     # slides, report, images
  |___ evaluation               # evaluation on both original and modified predictions
    |____ openpifpaf                  # evaluation for the original predictions
    |____ recent-th-70.0-mr-10-ml-5   # evaluation for the modified predictions
  |___ posetrack2018_modified   # modified predictions for posetrack2018 validation set
    |____ recent-th-70.0-mr-10-ml-5 
  |___ posetrack2018_openpifpaf # original predictions from openpifpaf for posetrack2018 validation set
  |___ reid_model               # storing deep re-ID model
  |___ storage                  # storage files containing instance of Storage class for each sequence
  |___ track_recover            # source code
  |___ ... 
```

The evaluation should be in [evaluation](./evaluation) folder. To have the official evaluation benchmark for Posetrack2018, refer to the [poseval](https://github.com/svenkreiss/poseval) evaluation package.

### Acknowledgements

The `FeatureExtractor` is adapted from [Torchreid](https://github.com/KaiyangZhou/deep-person-reid).

The `trackval` module is extended from [poseval](https://github.com/svenkreiss/poseval).

The initial idea was inspired by this [blog](https://towardsdatascience.com/fall-detection-using-pose-estimation-a8f7fd77081d).
