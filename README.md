# FEGAN: A High-performance Font Enhancement Network for Text CAPTCHA Preprocessing
This repository provides the PyTorch implementation of FEGAN.



To install requirements:

```setup
conda env create -n [your env name] -f environment.yaml
conda activate [your env name]
```

## To train the model
1. If traindata is the root directory of the training dataset, put your clean and noisy data in the train/clean and train/noisy folders, respectively
```
traindata
└─── noisy
└─── clean
```
2. After that, run this command:

```
python main.py --test 0 --noisy_src_path traindata/noisy --clean_src_path traindata/clean 
```
3. **Run the code below directly!**  Since we placed some training, validation and testing images in the data folder of the project root directory, you can directly try the example code:

```
python main.py --test 0 --noisy_src_path data/train/train_clean_100 --clean_src_path data\train\train_clean_100 
```



## To evaluate the model

### To validate
1. If valdata is the root directory of the validation dataset, put your clean and noisy data in the valdata/clean and valdata/noisy folders, respectively
```
valdata
└─── noisy
└─── clean
```
2. After that, run this command:

```
python main.py --test 1 --datarootN_val valdata
```

### To test 
1. If testdata is the root directory of the testing dataset, put noisy data in the testdata/noisy folders, respectively
```
testdata
└─── noisy
```
2. After that, run this command:

```
python main.py --test 1 --datarootN_val testdata --testmodel_n2c your_model
```

3. Run the code below directly! Since we provided pre-trained models and dataset samples, you can try this code.

```
python main.py --test 2 --datarootN_val data/test_mcaptcha --testmodel_n2c checkpoints/last-mcaptcha2.pth
```



## Pre-trained model

We provide pre-trained models in `./checkpoints` directory. You can download M-CAPTCHA [here](https://www.kaggle.com/datasets/sanluo/mcaptcha).  You can use `pip install captcha` to install the captcha library and generate P-CAPTCHA with only uppercase letters using the default settings.
```
checkpoints
|   AWGN_sigma15.pth # pre-trained model on M-CAPTCHA
|   AWGN_sigma25.pth # pre-trained model on P-CAPTCHA 
```

## Acknowledgements
This code is built on [UID-FDK](https://github.com/jdg900/UID-FDK). We thank the authors for sharing their codes.


## Contact
If you have any questions, feel free to contact me (wanxing123321@gmai.com)