# Vision_BGV_Box_Gate_Vector_Simulator
> A Logic Gate Simulator like Circuit Verse, Academo.org etc. which support's Vision mode. And can easily  be controlled using hand gestures.

## How To Run
1. ### If you are using anaconda
    * Go to search and find Anaconda Prompt
    * Change envirenment if needed 
        ```
        (base) C:\Users\user_name>activate env_name
        ```
    * Using cd for going to unzip folder
    * Run `editor.bat`
        ```
        (test) C:\Users\user_name>editor.bat
        ```
2. ### if you are using Python
    * Go to folder where you have unziped the file
    * Right click and click on `open on terminal`
    * Run `editor.bat`
        ```
        PS C:\Users\user_name\path\to\unzip\folder> .\editor.bat
        ```
    * If you are getting this error, it is because of envirenment paths
        ```
        'pip' is not recognized as an internal or external command
        ```
3. ### Run maunally
    * Use any IDE like. Visual Studio Code, PyCharm, Spyder etc.
    * Open the unzip folder
    * Install required packages
    * Run `main.py`
## Table of Contents
```
Vision BGV Box Gate Vector
    |-- fonts (fonts used)
    |     |-- Queensider.ttf
    |     |-- QueensiderLight-ZVj3l.ttf
    |     |-- QueensiderMedium.ttf
    |   
    |-- gateRes (assets)
    |-- model (classifiaction model weights & sounds)
    |     |-- click.wav
    |     |-- model_cpu.pk
    |     |-- model_cuda.pk
    |     |-- simulate.wav
    |
    |-- Model Recipe (For Custom Classification)
    |     |-- app.py
    |     |-- gesture_recognizer.py
    |     |-- model.ipynb
    |     |-- README.md
    |
    |-- Sample (photos and video)
    |-- saved (saved files)
    |-- shells (shell base)
    |-- singltion (widgets)
    |-- BSV.py (Architecture)
    |-- editor.bat (app runner)
    |-- gesture_recognizer.py (Gusture Class)
    |-- gui (gui related classes)
    |-- main.py (starting file)
    |-- README.md
    |-- requirements.txt (lib. names)
    |-- util_function (functions)
    |-- vectors.py (vector2D class)
```
### Architecture of Simulator 👇

![](Sample/arch_1.jpg)
## Samples
### What is what 👇
![](Sample/tutorial_1.png)
### Example of use 👇

https://github.com/Rajatsingh24/Vision_BGV_Box_Gate_Vector_Simulator/assets/114977647/b03e875f-d520-4087-8ce5-a2cb35b122f2

## How to Use:
1. ### Keys to control Simulator 
    * For menu : `Shift + RMB`
    * For Select gate/connection : `LMB`
    * For Toggle Input : `Ctrl + RMB`
    * For Deselect connection : `RMB`
2. ### Included Gesture in Vision Mode
$~~~~~~~~~~~~~~~~~~~~~~~$ <img src="https://github.com/Rajatsingh24/Vision_BGV_Box_Gate_Vector_Simulator/blob/main/Sample/6.jpg?raw=true" alt="drawing" width="100"/>

$~~~~~~~~~~~~~~~~~~~~~~~~~~$ `Recenter`

$~~~~~~~~~~~~~~~~~~~~~~~$ <img src="https://github.com/Rajatsingh24/Vision_BGV_Box_Gate_Vector_Simulator/blob/main/Sample/1.jpg?raw=true" alt="drawing" width="100"/>

$~~~~$ `Preffered Gesture for mouse pointer`

$~~~~~~~~~~~~~~~~~~~~~~~$ <img src="https://github.com/Rajatsingh24/Vision_BGV_Box_Gate_Vector_Simulator/blob/main/Sample/2.jpg?raw=true" alt="drawing" width="100"/>

$~~~~~~~~~~~~~~~~~~~$ `Mouse Left Click`

$~~~~~~~~~~~~~~~~~~~~~~~$ <img src="https://github.com/Rajatsingh24/Vision_BGV_Box_Gate_Vector_Simulator/blob/main/Sample/3.jpg?raw=true" alt="drawing" width="100"/>

$~~~~~~~~~~~~~~~~~~~$ `Mouse Right Click`

$~~~~~~~~~~~~~~~~~~~~~~~$ <img src="https://github.com/Rajatsingh24/Vision_BGV_Box_Gate_Vector_Simulator/blob/main/Sample/4.jpg?raw=true" alt="drawing" width="100"/>

$~~~~~~~~~~~~~~~~~~~~~~~~~~~$ `Menu Bar`

$~~~~~~~~~~~~~~~~~~~~~~~$ <img src="https://github.com/Rajatsingh24/Vision_BGV_Box_Gate_Vector_Simulator/blob/main/Sample/5.jpg?raw=true" alt="drawing" width="100"/>

$~~~~~~~~~~~~~~~~~~~~~~~$ `Toggle Imput`

$~~~~~~~~~~~~~~~~~~~~~~~$ <img src="https://github.com/Rajatsingh24/Vision_BGV_Box_Gate_Vector_Simulator/blob/main/Sample/7.jpg?raw=true" alt="drawing" width="100"/>

$~~~~~~~~~~~~~~~~~~~~~~~~~~$ `Simulate`


## Created BY :
    
### *Gaurav*, *Rajat Singh*

    
[Gaurav GitHub,](https://github.com/green-gray-gaurav "gitHub link") [Rajat GitHub](https://github.com/Rajatsingh24 "gitHub link")
