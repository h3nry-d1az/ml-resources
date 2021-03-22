#!/usr/bin/env python
import pyttsx3 as pt
import PyPDF2  as pp

from sys import argv


books = {
    # Books
    'The official Python 3 tutorial': 'The official Python 3 tutorial.pdf',
    'An introduction to the R language': 'An introduction to the R language.pdf',
    'An introduction to neural networks for beginners': 'An introduction to neural networks for beginners.pdf',
    'The official PyTorch tutorial': 'The official PyTorch tutorial.pdf',
    'Tutorialspoint TensorFlow tutorial': 'Tutorialspoint TensorFlow tutorial.pdf',
    'Natural Language Processing with Python': 'Natural Language Processing with Python.pdf',
    'scikit-learn official manual': 'scikit-learn official manual.pdf',
    'Pattern Recognition and Machine Learning': 'Pattern Recognition and Machine Learning.pdf',
    # Articles
    'DeepMind AlphaFold': 'DeepMind AlphaFold.pdf',
    'Go-Explore RL Algorithm': 'Go-Explore RL Algorithm.pdf',
    'Pix2Pix Image Translation': 'Pix2Pix Image Translation.pdf',
    'MeInGame: Portrait to Game Face': 'MeInGame Portrait to Game Face.pdf',
    'NeRF: Scenes as Neural Radiance Fields': 'NeRF Scenes as Neural Radiance Fields.pdf',
    'NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections': 'NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections.pdf',
    'CycleGAN: Image-to-Image Translation using Cycle-Consistent Adversarial Networks': 'CycleGAN: Image-to-Image Translation using Cycle-Consistent Adversarial Networks.pdf',
    'StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks': 'StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks.pdf',
    'GPT-3: Language Models are Few-Shot Learners': 'GPT-3: Language Models are Few-Shot Learners.pdf',
    'BlenderBot: Recipes for building an open-domain chatbot': 'BlenderBot: Recipes for building an open-domain chatbot.pdf',
    'Unsupervised Translation of Programming Languages': 'Unsupervised Translation of Programming Languages.pdf',
    'Playing Atari with Deep Reinforcement Learning': 'Playing Atari with Deep Reinforcement Learning.pdf',
    'DeepMind AlphaZero': 'DeepMind AlphaZero.pdf',
    'Learning to Simulate Dynamic Environments with GameGAN': 'Learning to Simulate Dynamic Environments with GameGAN.pdf',
    'TensorFlow Quantum': 'TensorFlow Quantum.pdf'
}

def usage():
    print ('HDB-PROGRAMMING\'s book reader - audio.py')
    print ('Usage: audio.py [book name]')
    print ('')
    print ('Example: audio.py NeRF')


if len(argv) < 2:
    usage()
elif argv[1] == '--booklist':
    for b in books:
        print(f'{b} ======> {books[b]}')
else:
    try:
        book = open(books[argv[1]], 'rb')
    except Exception as e:
        print(f'Book {argv[1]} could not be found')
        print('----------------')
        print('Remember to input just the book title (don\'t include the .pdf extension)')
        print('Also remember to input just books which are in PDF format (LaTeX or PostScript not valid)')

    pdfReader = pp.PdfFileReader(book)
    pages = pdfReader.numPages

    speaker = pt.init()
    speaker.setProperty('rate', 130)

    for num in range(pages):
        page = pdfReader.getPage(num)
        text = page.extractText()
        speaker.say(text)
        speaker.runAndWait()