#!/usr/bin/env python
import pyttsx3 as pt
import PyPDF2  as pp

from sys import argv


books = {
    'NeRF': 'NeRF Representing Scenes as Neural Radiance Fields for View Synthesis.pdf'
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
        book = open(books[argv[1]])
    except Exception as e:
        print(f'Book {argv[1]} could not be found')
        print('----------------')
        print('Remember to input just the book title (don\'t include the .pdf extension)')
        print('Also remember to input just books which are in PDF format (LaTeX or PostScript not valid)')

    pdfReader = pp.PdfFileReader(book)
    pages = pdfReader.numPages

    speaker = pt.init()
    speaker.setProperty('rate', 125)

    for num in range(pages):
        page = pdfReader.getPage(num)
        text = page.extractText()
        speaker.say(text)
        speaker.runAndWait()