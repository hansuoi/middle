def make_title(title):
    imsdb_title = ''
    for i in range(len(title)):
        if title[i]=='\n' or title[i]==':':
            continue
        elif title[i] == ' ':
            imsdb_title += '-'
        else:
            imsdb_title += title[i]
    IMSDb_url = 'https://www.imsdb.com/scripts/' + imsdb_title + '.html'

    return IMSDb_url


import requests
from bs4 import BeautifulSoup
import time
import re

with open('./title/add_title.txt', mode='r', encoding='utf-8') as f:
    title = f.read().split('\n')

names = ''
counter = 1

for t in title:
    d = {}

    IMSDb_url = make_title(t)
    url = requests.get(IMSDb_url)
    soup = BeautifulSoup(url.content, "html.parser")
    b = soup.findAll("b")

    for i in range(len(b)):
        name = re.findall('([^\s].*)', b[i].text)
        if len(name) == 0:
            continue
        else:
            for j in range(len(name[0])-1):
                if name[0][j]=='(' or (name[0][j]==' ' and name[0][j+1]==' '):
                    name[0] = name[0][:j]
                    break
            if len(name[0])-1>=0 and name[0][len(name[0])-1] == ' ':
                name[0] = name[0][:len(name[0])-1]
        if name[0] in d:
            d[name[0]] += 1
        else:
            d[name[0]] = 1

    if len(d) != 0:
        print('No.', counter, ' ', t)

        sort_d = sorted(d.items(), key=lambda x:x[1], reverse=True)
        if len(sort_d) > 6:
            mini_sort_d = list(sort_d)[:5]
            print_d = {}
            for m in mini_sort_d:
                print_d[m[0]] = m[1]
            print(print_d, end='')
            print(', ...')
        else:
            print(sort_d)

        while len(d) != 0:
            hero_name = max(d, key=d.get)
            print('name = ', hero_name)
            while True:
                print('OK? y/n/c')
                ans = input()
                if ans=='y' or ans=='n' or ans=='c':
                    break
            if ans == 'y':
                names += hero_name
                break
            elif ans == 'c':
                break
            elif ans == 'n':
                d.pop(hero_name)
                print('\n')
    names += '\n'
    counter += 1
    print('\n')

    time.sleep(1)


with open('./title/hero_name.txt', mode='w', encoding='utf-8') as f:
    f.writelines(names)


print('Done!')
