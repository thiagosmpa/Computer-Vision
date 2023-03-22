# UNET Image Segmentation Using PyTorch
## Dataset - Digital Retinal Images for Vessel Extraction
- Dataset found at: https://www.kaggle.com/datasets/zionfuo/drive2004
- Credits: https://www.youtube.com/watch?v=T0BiFBaMLDQ <br />
- Special thanks to the teacher from the youtube channel "Idiot Developer"!

Python Version - Python 3.9.16
<br />
<br />
The dataset contain 40 digitalized retina images, divided in Traning set and Test set. <br />
The training set contains three directories: "1st_manual", "mask" and "images", as for the test data, it contains four directories: "1st_manual", "2nd_manual", "mask" and "images". <br />
<br />
For this task, we won't need the directories: "mask" from both training and test set, neither "2nd_manual" from test set. This is because only the "1st_manual" will work as the real mask for our images. In this case you will obtain only the images:

![21_training](https://user-images.githubusercontent.com/33949962/226976601-df3d8538-a4f1-4958-adf4-e7d34392b850.png)

![21_manual1](https://user-images.githubusercontent.com/33949962/226976350-6c2fc164-8faa-4c3b-82bb-8a47827bc41e.gif)


## Pre Processing

Before runing the code, make sure that the dataset directories is correct.
```
""" Load the data """
data_path = './dataset/'
```
The data_path must contain the directory that contains the two directories: "traning" and "test". <br />
After that, you can also choose whether you want to augment the data or not. <br /> <br />
**Data Augmentation** is used to create a wider traning dataset, by creating images with modifications of the originals, like:
- Rotation
- Vertical Flipe 
- Horizontal Flip

But if your machine doesn't have GPU aceleration for the training process, you can choose not to augment data in order to use less data and save some machine overload.
```
""" Data augmentation """
augment_data(train_x, train_y, "new_data/train/", augment=False)
augment_data(test_x, test_y, "new_data/test/", augment=False)
```

Observation: Data Augmentation is used only in the training process. It doesn't need to augment test data. So choose between "False" or "True" only in the first line. <br /> <br />

## Traning Process
After runing correctly the pre processing script, you can run "train.py" in order to train the model.

To see if your machine has or not an available CUDA for training with the GPU, you can run:
```
print(torch.cuda.device_count())
```
This will show how many GPUs you have to alocate the process in. <br /> <br />

Also, if you want to continue the previous training, you can uncomment the statement:
```
""" Transfer Learning Part """
if os.path.isfile(checkpoint_path):
    print('Loading checkpoint...')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
```
This will load the model by the best checkpoint and keep training from that. If you want to train from the beggining, leave that commented. <br />
For Fine Tuning, you can choose the Hyperparameters in:
```
""" Hyperparameters """
    batch_size = 2
    num_epochs = 50
    lr = 1e-4
```
Which the important ones are the **num_epochs** that stands for the number of epochs and **lr** which stands for the learning rate. <br />
increasing the number of epochs will increase how many times the model will be training and <br />
modifying the learning rate will increase how fast will the coefficients will increase or decrease. In other words, the loss will decrease faster, but can occur to loose the minimum error - miss the global minimum. <br /> <br />

## Validation
Runing the "test.py" will create a directory with the results and run the evaluation metrics for the model. To evaluate the model, this script will load the checkpoints saved from the training process. <br />
In this case, the evaluation metrics used were: <br />
-   jaccard, f1_score, recall, precision and accuracy. <br />
After training the data for 50 epochs, learning rate 1e-4, the obtained metrics were:

<img width="567" alt="Captura de Tela 2023-03-22 às 13 35 47" src="https://user-images.githubusercontent.com/33949962/226974864-8f694fb6-f43f-4714-bd76-b28ea4f7e2d9.png">

Resulting in the following image, which contains the Original Image, Original Mask and Predicted Mask.
![01_test_0](https://user-images.githubusercontent.com/33949962/226976110-bf39b330-e67d-4327-9e67-37b7784dfa54.png)

