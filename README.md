# Automated Checks for Image Annotation Quality
Image Annotations have been checked programmatically using this code. Different checks are implemented to check different error sources. 4 different checks have been used in this code as of now.
More robust checks can be added in the future and the different parameters involved in the current 4 checks can be tuned to make them better
A final probability sum is calculated by adding each of the error probabilities from each check. 
Depending on the final probability value the image with its current annotations is classified into three categories  - All Good (if everything seems ok), Warning (if there could be an error), Error (if there is has high chance of error). Finally for each image the error classifications and final probability values are stored in a csv file 

### Please add your image urls and keys to the .py file as dummy values have been used in this code
### To run the code, after adding the urls and keys just type python image annotationchecks.py on a code editor that supports python and has the various python dependencies and modules installed 
### In case you find this repository useful please star it and in case you want to discuss any improvements or so to this code please let me know

# Results Output on Terminal
![image](https://github.com/user-attachments/assets/cabf0dbe-4873-47c4-8da8-d3563e7ffe85)

![image](https://github.com/user-attachments/assets/ddc38e65-bee3-406d-a517-5790db79a885)

