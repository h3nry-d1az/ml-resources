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
    'Pattern Recognition and Machine Learning': 'https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf',
    # Articles
    'DeepMind AlphaFold': 'https://arxiv.org/pdf/1911.05531.pdf',
    'Go-Explore RL Algorithm': 'https://arxiv.org/pdf/1901.10995.pdf',
    'Pix2Pix Image Translation': 'https://arxiv.org/pdf/1611.07004.pdf',
    'MeInGame: Portrait to Game Face': 'https://arxiv.org/pdf/2102.02371.pdf',
    'NeRF: Scenes as Neural Radiance Fields': 'https://arxiv.org/pdf/2003.08934.pdf',
    'NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections': 'https://arxiv.org/pdf/2008.02268.pdf',
    'CycleGAN: Image-to-Image Translation using Cycle-Consistent Adversarial Networks': 'https://arxiv.org/pdf/1703.10593.pdf',
    'StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks': 'https://arxiv.org/pdf/1812.04948.pdf',
    'GPT-3: Language Models are Few-Shot Learner': 'https://arxiv.org/pdf/2005.14165.pdf',
    'BlenderBot: Recipes for building an open-domain chatbot': 'https://arxiv.org/pdf/2004.13637.pdf',
    'Unsupervised Translation of Programming Languages': 'https://arxiv.org/pdf/2006.03511.pdf',
    'Playing Atari with Deep Reinforcement Learning': 'https://arxiv.org/pdf/1312.5602.pdf',
    'DeepMind AlphaZero': 'https://arxiv.org/pdf/1712.01815.pdf',
    'Learning to Simulate Dynamic Environments with GameGAN': 'https://arxiv.org/pdf/2005.12126.pdf',
    'TensorFlow Quantum': 'https://arxiv.org/pdf/2003.02989.pdf',
    'Analyzing and Improving the Image Quality of StyleGAN': 'https://arxiv.org/pdf/1912.04958.pdf',
    'Learning to Simulate Complex Physics with Graph Networks': 'https://arxiv.org/pdf/2002.09405.pdf',
    'Stanza: A Python Natural Language Processing Toolkit for Many Human Languages': 'https://arxiv.org/pdf/2003.07082.pdf',
    'YOLOv4: Optimal Speed and Accuracy of Object Detection': 'https://arxiv.org/pdf/2004.10934.pdf',
    'ResNet: Deep Residual Learning for Image Recognition': 'https://arxiv.org/pdf/1512.03385.pdf',
    'PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models': 'https://arxiv.org/pdf/2003.03808.pdf',
    'Efficient Estimation of Word Representations in Vector Space': 'https://arxiv.org/pdf/1301.3781.pdf'
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