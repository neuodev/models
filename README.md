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

## Development Process

After generating an **idea** (like: `Arabic-English` translator) and getting the right dataset

1. Start exploring the data in my local machine
2. Build _basic model_ on a subset of the dataset. I tune the hyperparameters until I start getting promising results
3. Push the code to GitHub and clone it to my server in which I can run **intensive** training for the current model.
4. I run the training as a process using `PM2` library, leave it until it done traning.
5. `train.py` always saves the model into the same directory as `model.h5` also the history as `history.json`
6. I push the code from my server to GitHub, I clone it to my local to perform some perdictions on the model and to visualize the **learning curve**
7. I repate the whole process again with more tuning to the hyperparameters
8. I have a background in **web development** so I may expand the idea later to build a web app that put this model into **production**
   I am using `Django` or `Flask` as my backend and `Reactjs` ad my frontend. I use `tailwindcss` for styling

```bash
My MacBook ------------------> GitHub --------------------------> My Server
   ^                          |   ^                                     |
   |                          |   |                                     |
   |                          V   |                                     |
   ----<--------<----------<--<---^----<-------<-----------------------<-
```

## Process On The Server

Starting from cloning the latest code on GitHub I am runing the traning process on the server
Normally You will connect to the server using the `Terminal`. Something you will notice that if you closed the
terminal the traning process will stop!!. A perfect solution to this is to run the traning as a `process` on the server
this means that the traning will run on the backend and continue running even if you closed the terminal.
A great tool for this is [PM2](https://pm2.keymetrics.io/docs/usage/quick-start/). It jsut works!!
it is a `npm` package you need to have `nodejs` and `npm` installed in your server
You install it by..

```bash
npm install pm2@latest -g
```

You will be able to use it from the terminal

To perform a traning

```bash
pm2 start "python3 train.py"
```

And It will start the traning on the backend
To see the current processes just type

```bash
pm2 list
```

To see the logs of the traning process

```bash
pm2 logs
```

# Server

I dont' need a crazy server to run the traning. I am using a server from **Digital ocean** to berform the traning
and it has this

1.  8 GB Memory
2.  4 Intel vCPUs
3.  160 GB Disk
4.  based on NYC1 city
5.  Running Ubuntu 20.04 (LTS) x64

I can leave it on training the hole night!!
