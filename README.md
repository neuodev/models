# Models

## Why did created this repo?

To make a realy pwerful models you need to run them on some **server** I have my own server running on _Digital Ocean_.
So I develop the model in my local enviornment and train the model on a sample of the dataset then push the code
pull it into my server and run the model as a process on the backend. This allow me to run the traning process to a long period of time. And try large number of **epochs**, **batch sizes**, apply **grid search** and use **large dataset**

> All of this to **unleash** the power of **deep leaning** and get **results**

## Architecture of directores

```bash
root
|
|----- project title
|      |
|      |---- train.py # Process the dataset and train the model
|      |---- model.h5 # `train.py` will save the model in h5 formate so I can use it locally in my macine
|      |----- history.json # `train.py` will save the training histroy into a jons file to visualize it after the training is done
```
