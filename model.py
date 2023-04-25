import os
import shutil
import time
import glob
import torch
from PIL import Image
import cv2
import random
import string
import numpy as np
import random

def get_random_string(length):
    """
    Generate a random string of fixed length 

    Inputs
    ------
    length: int - length of the string to be generated

    Returns
    -------
    str - random string

    """
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    return result_str

def load_model():
    """
    Load the model from the local directory
    """
    #model = torch.hub.load('./', 'custom', path='YOLOv5_new.pt', source='local')
    model = torch.hub.load('./', 'custom', path='Week_9.pt', source='local')
    return model

def draw_own_bbox(img,x1,y1,x2,y2,label,color=(36,255,12),text_color=(0,0,0)):
    """
    Draw bounding box on the image with text label and save both the raw and annotated image in the 'own_results' folder

    Inputs
    ------
    img: numpy.ndarray - image on which the bounding box is to be drawn

    x1: int - x coordinate of the top left corner of the bounding box

    y1: int - y coordinate of the top left corner of the bounding box

    x2: int - x coordinate of the bottom right corner of the bounding box

    y2: int - y coordinate of the bottom right corner of the bounding box

    label: str - label to be written on the bounding box

    color: tuple - color of the bounding box

    text_color: tuple - color of the text label

    Returns
    -------
    None

    """
    name_to_id = {
        "NA": 'NA',
        "Bullseye": 10,
        "One": 11,
        "Two": 12,
        "Three": 13,
        "Four": 14,
        "Five": 15,
        "Six": 16,
        "Seven": 17,
        "Eight": 18,
        "Nine": 19,
        "A": 20,
        "B": 21,
        "C": 22,
        "D": 23,
        "E": 24,
        "F": 25,
        "G": 26,
        "H": 27,
        "S": 28,
        "T": 29,
        "U": 30,
        "V": 31,
        "W": 32,
        "X": 33,
        "Y": 34,
        "Z": 35,
        "Up": 36,
        "Down": 37,
        "Right": 38,
        "Left": 39,
        "Up Arrow": 36,
        "Down Arrow": 37,
        "Right Arrow": 38,
        "Left Arrow": 39,
        "Stop": 40
    }
    # Reformat the label to {label name}-{label id}
    label = label + "-" + str(name_to_id[label])
    # Convert the coordinates to int
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    # Create a random string to be used as the suffix for the image name, just in case the same name is accidentally used
    rand = str(int(time.time()))

    # Save the raw image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"own_results/raw_image_{label}_{rand}.jpg", img)

    # Draw the bounding box
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # For the text background, find space required by the text so that we can put a background with that amount of width.
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    # Print the text  
    img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    img = cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
    # Save the annotated image
    cv2.imwrite(f"own_results/annotated_image_{label}_{rand}.jpg", img)


def predict_image(image, model, signal):
    """
    Predict the image using the model and save the results in the 'runs' folder
    
    Inputs
    ------
    image: str - name of the image file

    model: torch.hub.load - model to be used for prediction

    signal: str - signal to be used for filtering the predictions

    Returns
    -------
    str - predicted label
    """
    try:
        # Load the image
        img = Image.open(os.path.join('uploads', image))

        # Predict the image using the model
        results = model(img)

        # Images with predicted bounding boxes are saved in the runs folder
        results.save('runs')

        # Convert the results to a pandas dataframe and calculate the height and width of the bounding box and the area of the bounding box
        df_results = results.pandas().xyxy[0]
        df_results['bboxHt'] = df_results['ymax'] - df_results['ymin']
        df_results['bboxWt'] = df_results['xmax'] - df_results['xmin']
        df_results['bboxArea'] = df_results['bboxHt'] * df_results['bboxWt']

        # Label with largest bbox height will be last
        df_results = df_results.sort_values('bboxArea', ascending=False)

        # Filter out Bullseye
        pred_list = df_results 
        pred_list = pred_list[pred_list['name'] != 'Bullseye']
        
        # Initialize prediction to NA
        pred = 'NA'

        # Ignore Bullseye unless they are the only image detected and select the last label in the list (the last label will be the one with the largest bbox height)
        if len(pred_list) == 1:
            if pred_list.iloc[0]['name'] != 'Bullseye':
                pred = pred_list.iloc[0]

        # If more than 1 label is detected
        elif len(pred_list) > 1:

            # More than 1 Symbol detected, filter by confidence and area
            pred_shortlist = []
            current_area = pred_list.iloc[0]['bboxArea']
            # For each prediction, check if the confidence is greater than 0.5 and if the area is greater than 80% of the current area or 60% if the prediction is 'One'
            for _, row in pred_list.iterrows():
                if row['name'] != 'Bullseye' and row['confidence'] > 0.5 and ((current_area * 0.8 <= row['bboxArea']) or (row['name'] == 'One' and current_area * 0.6 <= row['bboxArea'])):
                    # Add the prediction to the shortlist
                    pred_shortlist.append(row)
                    # Update the current area to the area of the prediction
                    current_area = row['bboxArea']
            
            # If only 1 prediction remains after filtering by confidence and area
            if len(pred_shortlist) == 1:
                # Choose that prediction
                pred = pred_shortlist[0]

            # If multiple predictions remain after filtering by confidence and area
            else:
                # Use signal of {signal} to filter further 
                
                # Sort the predictions by xmin
                pred_shortlist.sort(key=lambda x: x['xmin'])

                # If signal is 'L', choose the first prediction in the list, i.e. leftmost in the image
                if signal == 'L':
                    pred = pred_shortlist[0]
                
                # If signal is 'R', choose the last prediction in the list, i.e. rightmost in the image
                elif signal == 'R':
                    pred = pred_shortlist[-1]
                
                # If signal is 'C', choose the prediction that is central in the image
                else:
                    # Loop through the predictions shortlist
                    for i in range(len(pred_shortlist)):
                        # If the xmin of the prediction is between 250 and 774, i.e. the center of the image, choose that prediction
                        if pred_shortlist[i]['xmin'] > 250 and pred_shortlist[i]['xmin'] < 774:
                            pred = pred_shortlist[i]
                            break
                    
                    # If no prediction is central, choose the one with the largest area
                    if isinstance(pred,str):
                        # Choosing one with largest area if none are central
                        pred_shortlist.sort(key=lambda x: x['bboxArea']) 
                        pred = pred_shortlist[-1]
        
        # Draw the bounding box on the image
        if not isinstance(pred,str):
            draw_own_bbox(np.array(img), pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'], pred['name'])

        name_to_id = {
            "NA": 'NA',
            "Bullseye": 10,
            "One": 11,
            "Two": 12,
            "Three": 13,
            "Four": 14,
            "Five": 15,
            "Six": 16,
            "Seven": 17,
            "Eight": 18,
            "Nine": 19,
            "A": 20,
            "B": 21,
            "C": 22,
            "D": 23,
            "E": 24,
            "F": 25,
            "G": 26,
            "H": 27,
            "S": 28,
            "T": 29,
            "U": 30,
            "V": 31,
            "W": 32,
            "X": 33,
            "Y": 34,
            "Z": 35,
            "Up": 36,
            "Down": 37,
            "Right": 38,
            "Left": 39,
            "Up Arrow": 36,
            "Down Arrow": 37,
            "Right Arrow": 38,
            "Left Arrow": 39,
            "Stop": 40
        }
        # If pred is not a string, i.e. a prediction was made and pred is not 'NA'
        if not isinstance(pred,str):
            image_id = str(name_to_id[pred['name']])
        else:
            image_id = 'NA'
        print(f"Final result: {image_id}")
        return image_id
    # If some error happened, we just return 'NA' so that the inference loop is closed
    except:
        print(f"Final result: NA")
        return 'NA'

def predict_image_week_9(image, model):
    # Load the image
    img = Image.open(os.path.join('uploads', image))
    # Run inference
    results = model(img)
    # Save the results
    results.save('runs')
    # Convert the results to a dataframe
    df_results = results.pandas().xyxy[0]
    # Calculate the height and width of the bounding box and the area of the bounding box
    df_results['bboxHt'] = df_results['ymax'] - df_results['ymin']
    df_results['bboxWt'] = df_results['xmax'] - df_results['xmin']
    df_results['bboxArea'] = df_results['bboxHt'] * df_results['bboxWt']

    # Label with largest bbox height will be last
    df_results = df_results.sort_values('bboxArea', ascending=False)
    pred_list = df_results 
    pred = 'NA'
    # If prediction list is not empty
    if pred_list.size != 0:
        # Go through the predictions, and choose the first one with confidence > 0.5
        for _, row in pred_list.iterrows():
            if row['name'] != 'Bullseye' and row['confidence'] > 0.5:
                pred = row    
                break

        # Draw the bounding box on the image 
        if not isinstance(pred,str):
            draw_own_bbox(np.array(img), pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'], pred['name'])
        
    # Dictionary is shorter as only two symbols, left and right are needed
    name_to_id = {
        "NA": 'NA',
        "Bullseye": 10,
        "Right": 38,
        "Left": 39,
        "Right Arrow": 38,
        "Left Arrow": 39,
    }
    # Return the image id
    if not isinstance(pred,str):
        image_id = str(name_to_id[pred['name']])
    else:
        image_id = 'NA'
    return image_id


def stitch_image():
    """
    Stitches the images in the folder together and saves it into runs/stitched folder
    """
    # Initialize path to save stitched image
    imgFolder = 'runs'
    stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')

    # Find all files that ends with ".jpg" (this won't match the stitched images as we name them ".jpeg")
    imgPaths = glob.glob(os.path.join(imgFolder+"/detect/*/", "*.jpg"))
    # Open all images
    images = [Image.open(x) for x in imgPaths]
    # Get the width and height of each image
    width, height = zip(*(i.size for i in images))
    # Calculate the total width and max height of the stitched image, as we are stitching horizontally
    total_width = sum(width)
    max_height = max(height)
    stitchedImg = Image.new('RGB', (total_width, max_height))
    x_offset = 0

    # Stitch the images together
    for im in images:
        stitchedImg.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    # Save the stitched image to the path
    stitchedImg.save(stitchedPath)

    # Move original images to "originals" subdirectory
    for img in imgPaths:
        shutil.move(img, os.path.join(
            "runs", "originals", os.path.basename(img)))

    return stitchedImg

def stitch_image_own():
    """
    Stitches the images in the folder together and saves it into own_results folder

    Basically similar to stitch_image() but with different folder names and slightly different drawing of bounding boxes and text
    """
    imgFolder = 'own_results'
    stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')

    imgPaths = glob.glob(os.path.join(imgFolder+"/annotated_image_*.jpg"))
    imgTimestamps = [imgPath.split("_")[-1][:-4] for imgPath in imgPaths]
    
    sortedByTimeStampImages = sorted(zip(imgPaths, imgTimestamps), key=lambda x: x[1])

    images = [Image.open(x[0]) for x in sortedByTimeStampImages]
    width, height = zip(*(i.size for i in images))
    total_width = sum(width)
    max_height = max(height)
    stitchedImg = Image.new('RGB', (total_width, max_height))
    x_offset = 0

    for im in images:
        stitchedImg.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    stitchedImg.save(stitchedPath)

    return stitchedImg

