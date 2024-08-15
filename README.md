<br />
<p align="center">
  <img src="/images/Map.png" alt="Logo" height=150 >
  <h1 align="center">
    CZ3004/SC2079 Multidisciplinary Project - Algorithm API
  </h1>
</p>

# Overview
2024 August Update: Someone sent me the slides from the briefing of this semester, this repository, along with my other MDP-related ones, are entirely STILL reusable as far as I can see. SCSE can become CCDS but MDP is still MDP. As usual, retrain the YOLO model (or use something more recent la). Once again, that is a 1-day thing. If you are using these repositories and you don't have a functioning, fully-integrated system by end of Week 4, reconsider your life choices and your peer evaluations.

**2023 Semester 1 Update**: At least from what my juniors told me, this repository, along with my other MDP-related ones, are entirely reusuable. The only exception is that you will need to retrain the YOLO model since the fonts/colors were changed. That is a 1-day thing. If you are using these repositories and you don't have a functioning, fully-integrated system by end of Week 4, reconsider your life choices.

Y'all, if you are using this code, which apparently a LOT of y'all are, at least star this repo leh

This repository contains the code for the algorithm and image recognition inference component of the CZ3004/SC2079 Multidisciplinary Project. The repository is responsible for the following:

- Finding the shortest path from starting point to all the obstacles
- Performing inference on the images captured by the robot to identify the symbols
- Stitching the images together to form a summary of the results

## Setup

```bash
pip install -r requirements.txt
```

Start the server by

```bash
python main.py
```

The server will be running at `localhost:5000`

### Misc

- Raw images from Raspberry Pi are stored in the `uploads` folder.
- After calling the `image/` endpoint, the annotated image (with bounding box and label) is stored in the `runs` and `own_results` folder.
- After calling the `stitch/` endpoint, two stitched images using two different functions (for redundancy) are saved at `runs/stitched.jpg` and in the `own_results` folder.

### Primers - Constants and Parameters 

#### Direction of the robot (d)

* `NORTH` - `UP` - 0
* `EAST` - `RIGHT` - 2
* `SOUTH` - `DOWN` - 4
* `WEST` - `LEFT` 6

#### Parameters

* `EXPANDED_CELL` - Size of an expanded cell, normally set to just 1 unit, but expanding it to 1.5 or 2 will allow the robot to have more space to move around the obstacle at the cost of it being harder to find a shortest path. Useful to tweak if robot is banging into obstacles.
* `WIDTH` - Width of the area (in 10cm units)
* `HEIGHT` - Height of the area (in 10cm units)
* `ITERATIONS` - Number of iterations to run the algorithm for. Higher number of iterations will result in a more accurate shortest path, but will take longer to run. Useful to tweak if robot is not finding the shortest path.
* `TURN_RADIUS` - Number of units the robot turns. We set the turns to `3 * TURN_RADIUS, 1 * TURN_RADIUS` units. Can be tweaked in the algorithm
* `SAFE_COST` - Used to penalise the robot for moving too close to the obstacles. Currently set to `1000`. Take a look at `get_safe_cost` to tweak.
* `SCREENSHOT_COST` - Used to penalise the robot for taking pictures from a position that is not directly in front of the symbol. 

### API Endpoints:


##### 1. POST Request to /path:

Sample JSON request body:

```bash
{
    "obstacles":
    [
        {
            "x": 0,
            "y": 9,
            "id": 1,
            "d": 2
        },
        ...,
        {
            "x": 19,
            "y": 14,
            "id": 5,
            "d": 6
        }
    ]
}
```

Sample JSON response:

```{
    "data": {
        "commands": [
            "FR00",
            "FW10",
            "SNAP1",
            "FR00",
            "BW50",
            "FL00",
            "FW60",
            "SNAP2",
            ...,
            "FIN"
        ],
        "distance": 46.0,
        "path": [
            {
                "d": 0,
                "s": -1,
                "x": 1,
                "y": 1
            },
            {
                "d": 2,
                "s": -1,
                "x": 5,
                "y": 3
            },
            ...,
            {
                "d": 2,
                "s": -1,
                "x": 6,
                "y": 9
            },
        ]
    },
    "error": null
}
```

##### 2. POST Request to /image

The image is sent to the API as a file, thus no `base64` encoding required.

**Sample Request in Python**

```python3
response = requests.post(url, files={"file": (filename, image_data)})
```

- `image_data`: a `bytes` object

The API will then perform three operations:

1. Save the received file into the `/uploads` and `/own_results` folders.
2. Use the model to identify the image, save the results into the folders above.
3. Return the class name as a `json` response.

**Sample json response**

```json
{
  "image_id": "D",
  "obstacle_id": 1
}
```

Please note that the inference pipeline is different for Task 1 and Task 2, be sure to comment/uncomment the appropriate lines in `app.py` before running the API.

##### 3. POST Request to /stitch

This will trigger the `stitch_image` and `stitch_image_own` functions.

- Images found in the `run/` and `own_results` directory will be stitched together and saved separately, producing two stitched images. We have two functions for redundancy purposes. In case one fails, the other can still run.

# Disclaimer

I am not responsible for any errors, mishaps, or damages that may occur from using this code. Use at your own risk. This code is provided as-is, with no warranty of any kind. 

# Acknowledgements

I used Group 28's algorithm as a baseline, but improved it significantly. Edge cases that were previously not covered/handled are now handled.
- [Group 28](https://github.com/CZ3004-Group-28)

# Related Repositories

* [Website](https://github.com/pyesonekyaw/MDP-Showcase)
* [Simulator](https://github.com/pyesonekyaw/CZ3004-SC2079-MDP-Simulator)
* [Raspberry Pi](https://github.com/pyesonekyaw/CZ3004-SC2079-MDP-RaspberryPi)
* [Image Recognition](https://github.com/pyesonekyaw/CZ3004-SC2079-MDP-ImageRecognition)
