#!/usr/bin/env python
from urllib import request
from sys import argv


books = {
    # Books
    'The official Python 3 tutorial': 'https://bugs.python.org/file47781/Tutorial_EDIT.pdf',
    'An introduction to the R language': 'https://cran.r-project.org/doc/manuals/r-release/R-intro.pdf',
    'An introduction to neural networks for beginners': 'https://adventuresinmachinelearning.com/wp-content/uploads/2017/07/An-introduction-to-neural-networks-for-beginners.pdf',
    'The official PyTorch tutorial': 'https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf',
    'Tutorialspoint TensorFlow tutorial': 'https://www.tutorialspoint.com/tensorflow/tensorflow_tutorial.pdf',
    'Natural Language Processing with Python': 'http://www.datascienceassn.org/sites/default/files/Natural%20Language%20Processing%20with%20Python.pdf',
    'scikit-learn official manual': 'https://scikit-learn.org/0.18/_downloads/scikit-learn-docs.pdf',
    # Articles
    'DeepMind AlphaFold': 'https://arxiv.org/pdf/1911.05531.pdf',
    'Go-Explore RL Algorithm': 'https://arxiv.org/pdf/1901.10995.pdf',
    'Pix2Pix Image Translation': 'https://arxiv.org/pdf/1611.07004.pdf',
    'MeInGame: Portrait to Game Face': 'https://arxiv.org/pdf/2102.02371.pdf',
    'NeRF: Scenes as Neural Radiance Fields': 'https://arxiv.org/pdf/2003.08934.pdf',
    'NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections': 'https://arxiv.org/pdf/2008.02268.pdf',
    'CycleGAN: Image-to-Image Translation using Cycle-Consistent Adversarial Networks': 'https://arxiv.org/pdf/1703.10593.pdf',
    'StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks': 'https://arxiv.org/pdf/1812.04948.pdf'
}

def usage():
    print ('HDB-PROGRAMMING\'s book downloader - download.py')
    print ('Usage: download.py [book name]')
    print ('')
    print ('Example: download.py NeRF')


if len(argv) < 2:
    usage()
elif argv[1] == '--booklinks':
    for b in books:
        print(f'{b} ======> {books[b]}')
else:
    try:
        dbook = request.urlopen(books[argv[1]]).read()
    except Exception as e:
        print(f'Book {argv[1]} could not be found')
        print('----------------')
        print('Remember to input just the book title (don\'t include the .pdf extension)')

    try:
        book = open(f'{argv[1]}.pdf', 'wb')
        book.write(dbook)
        print('Book download finished!!!')
        print(f'Filename is {argv[1]}.pdf')
    except Exception as e:
        print(f'Cannot create {argv[1]}.pdf file')
        print(f'Caused by the following error:\n{e}')