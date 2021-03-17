#!/usr/bin/env python
from urllib import request
from sys import argv


books = {
    'NeRF': 'https://arxiv.org/pdf/2003.08934.pdf'
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