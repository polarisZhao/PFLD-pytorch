# PFLD-pytorch

Implementation of  PFLD A Practical Facial Landmark Detector by pytorch.

#### install requirements

~~~shell
pip3 install -r requirements.txt
~~~

#### Datasets

- **WFLW Dataset Download**

​    [Wider Facial Landmarks in-the-wild (WFLW)](https://wywu.github.io/projects/LAB/WFLW.html) is a new proposed face dataset. It contains 10000 faces (7500 for training and 2500 for testing)  with 98 fully manual annotated landmarks.

1. WFLW Training and Testing images [[Google Drive](https://drive.google.com/file/d/1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC/view?usp=sharing)] [[Baidu Drive](https://pan.baidu.com/s/1paoOpusuyafHY154lqXYrA)]
2. WFLW  [Face Annotations](https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz)
3. Unzip above two packages and put them on `./datasets/WFLW/`
4. move `Mirror98.txt` to `WFLW/WFLW_annotations`

~~~shell
$ cd data 
$ python3 SetPreparation.py
~~~

#### training & testing

training :

~~~shell
$ python3 train.py
~~~

testing:

~~~shell
$ python3 test.py
~~~

#### results:

![](./results/example.png)

#### reference: 

 PFLD: A Practical Facial Landmark Detector https://arxiv.org/pdf/1902.10859.pdf

#### TODO:

- [ ] fix bugs

- [ ] ncnn inference

- [ ] retrain on datasets AFLW and 300W

