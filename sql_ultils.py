#!/usr/bin/python
# -*- coding: utf-8 -*-

# author: Alexandre Robicquet
import MySQLdb as mdb
import urllib2
import time

htmlCodes = (
        ("'", '&#39;'),
        ('"', '&quot;'),
        ('>', '&gt;'),
        ('<', '&lt;'),
        ('&', '&amp;'),
        (' ','&#10;'),
        ("'",'â€™'),
        (' ','â€”'),
        ('é','Ã©'),
        ("'",'&apos;'),
        ("ø",'Ã,'),
        ('','User-contributed text is available under the Creative Commons By-SA License; additional terms may apply.'),
        ("'",'â€™')
)

alphabet = {'a' 'b' 'c' 'd' 'e' 'f' 'g' 'h' 'i' 'j'
'k' 'l' 'm' 'n' 'o' 'p' 'q' 'r' 's' 't' 'u' 'v' 'w' 'x' 'y' 'z'};

def extract(x,bw,ew):
    b = x.find(b)
    x_new = x[b:]
    e = x_new.find(ew)+b;

    w_extracted = x[b+len(b):e]
    retunr w_extracted
