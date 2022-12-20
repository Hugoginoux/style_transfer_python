# style_transfer_python

This algorithm uses transfer learning with vgg19 trained on image.net to merge the style of an image and the content of another. In order to run it :

1) Download the code with  

```console
foo@bar:~$ git clone https://github.com/Hugoginoux/style_transfer_python.git
```

2) Install the virtual environment with

```console
foo@bar:~$ pip install -r requirements.txt
```

3) Select 2 images, for instance the [Eiffel tower](https://sirmionebs.it/wp-content/uploads/2020/05/8x5-tour-eiffel-e1589350255566-826x516.jpg) for the content and a [Picasso](https://i.pinimg.com/originals/49/b9/73/49b9736a362a80564cd2ec54598f1bc4.png) for the style. Then run 

```console
foo@bar:~$ python main.py <path/to/content/image> <path/to/style/image>
```

The output should be stored in a "output.jpg"
