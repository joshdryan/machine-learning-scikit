Joshua Ryan

Description:
This machine learning project uses the MNIST handwritten digit database to learn and predict handwritten digits in an image format.

Dependencies:
generateClassifier.py, isolatedigits.py, performRecognition.py

Requirements:
- Python 3
- matplotlib
- Sklearn
- Skimage
- Scipy
- Numpy

Run as:
1. Generate classifier
$ python generateClassifier.py
2. Perform Recognition
$ python performRecognition.py

Operation:
First run generateClassifier.py, then run performRecognition.py. The program will show a before and after of the image in which digits are being identified. 
In order to change the file, specify new file on line 53 of performRecognition.py

Output:
The program uses matplotlib and outputs the image as a "before" and "after". In the "after" each recognized digit will have a box bordering it, and will have the predicted digit shows as well. 
