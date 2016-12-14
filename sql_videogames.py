#!/usr/bin/python
# -*- coding: utf-8 -*-

# author: Alexandre Robicquet
import MySQLdb as mdb
import urllib2
import time
from utils import *
urllib2.install_opener(urllib2.build_opener(urllib2.ProxyHandler({"http": 'http://'+random.choice(open('proxy.txt').readlines()).replace('\n','')})));


con=mdb.connect('localhost','root','user','test',charset='utf8')
urllib2.install_opener(urllib2.build_opener(urllib2.ProxyHandler()));

alphabet = {'a' 'b' 'c' 'd' 'e' 'f' 'g' 'h' 'i' 'j'
'k' 'l' 'm' 'n' 'o' 'p' 'q' 'r' 's' 't' 'u' 'v' 'w' 'x' 'y' 'z'};

url =
k=0;

for letter in alphabet:
    url = 'http://www.gamerevolution.com/game/all/'+letter+'/long_name/asc'
    opener = urllib2.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    response=opener.open(url)
    game_list = response.read()

    game_list = game_list[game_list.find('<tr class="trIndexList">'):]
    while game_list.find('<tr class="trIndexList">') != -1:

        e = game_list.find('</tr>');
        vg = game_list[:e];
        vg_url, vg = extract(vg,'<a href="','" class="')
        vg_name, vg = extract(vg,'class="headline">','</a></td>')

        game_list = game_list[e:]
        response=opener.open('http://www.gamerevolution.com/'+vg_url)
        vg_page = response.read()
