# VIZN: Illuminating Faces
### Colaborators
Jacob Mccalip, Daemon Henry Mutka, Julio cantu

## File list
- `README.md` - This file
- `.gitignore` - This file tells git which files to ignore.
- `.train` - Called to train and output a pkl of the trained data 
- `.testing` - Used along with a 1 or 2 to call either test1 or test 2 
- `.test1` - Basic training programs for the trained face detector
- `.test2` - Advanced traning programs for the trained face detector
- `.requirements` - Where the needed imports are located
- `.config.py` - Holds directory locations
- `.face_detection_model.pkl` - Facial detection model based off of just AdaBoosting
- `.face_detection_cascade.pkl` - Facial detection model using AdaBoost and cascade clasifiers
- `.sampleTestPyResults.txt` - Where the results are stored from the test programs
- `.testing.py (in src)` -
- `.skin_detection.py` - Used to create a(n) basic skin detection mask to use along with the trained model
- `.processing.py` -
- `.nms.py` -
- `.newSkin.py` - Used to create a(n) advanced skin mask using the UCI histogram
- `.model.py` - This file is holds the functions to save, load, or train the model
- `.cascade.py` - 
- `.boosting.py` - File used to cunduct the AdaBoosting technique 
- `.UCI_Skin_NonSkin.txt` - Histogram for skin and non-skin data
- `/important_outputs` - In the folder is where the basic and advanced data goes, along with the detected skin

## Setting up the environment

If you are running the code on your own machine, make sure python is installed and set up a virtual environment `venv`. make sure to also have `opencv` installed, you can do so you can use the `requirements.txt` file by using the following command in your terminal:

```bash
pip install -r requirements.txt
```

In vs code you can use `cmd + shift + p` to pull up the terminal to create a python enviornment, after chose `python create` then `venv` to finish setting up your enviornment run the command above in your terminal

## Running the code

You can run your code using the following commands:

The reposotory has a trained model, yet if you would like to train your own.
```bash
python train.py
```

## Running the tests

To run the tests make sure your in the main directory, not src, and run the following command:

For a(n) basic test
```bash
python testing.py 1
```

For a(n) advanced test
```bash
python testing.py 2
```

## Results

<p align="center">
Original Photo
</p>
<p align="center">
<img src="data/test_face_photos/DSC01181.JPG">
</p>


| Basic Skin Mask | Advanced Skin Mask | Basic Test | Advanced Test |
| :---: | :---: | :---: | :---: |
| <img src="important_outputs/basic_Skins/DSC01181.webp" width="250"> | <img src="important_outputs/skins/DSC01181.JPG" width="250"> | <img src="important_outputs/outputBasic/DSC01181.JPG" width="250"> | <img src="important_outputs/outputAdvanced/DSC01181.JPG" width="250"> |




