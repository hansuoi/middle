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





def pn_score(title=None, name=None):
    import pandas as pd
    import numpy as np
    import os
    import glob
    import pathlib
    import re
    import janome
    import jaconv
    import requests
    from bs4 import BeautifulSoup
    import treetaggerwrapper as ttw
    import warnings
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    warnings.simplefilter('ignore')

    posi_nega_df = pd.read_csv('dic/pn_en.dic.csv', index_col=0)

    if title == None:
        print('title = ', end='')
        title = input()
    if name == None:
        print('hero name = ', end='')
        name = input()

    IMSDb_url = make_title(title)
    url = requests.get(IMSDb_url)
    soup = BeautifulSoup(url.content, "html.parser")
    script = soup.find("b")
    b = soup.findAll("b")
    i = 1

    char_flag = False
    lines = []
    line = ''
    while script.next_sibling != None:
        script = script.next_sibling
        if script.string == b[i].text:
            i += 1
            if re.findall('^\s([^\r])*', script.string) != []:
                if re.findall('\s*([^\s]*)\s*', script.string)[0] == name:
                    char_flag = True
        else:
            s = script.string
            if char_flag:
                j = 0
                while len(s)>0 and s[0] == ' ':
                    s = s[1:]
                while len(s) > j+3  and  s[j] + s[j+1] + s[j+2] + s[j+3] != '\r\n\r\n':
                    line += s[j].replace('\n', ' ')
                    j += 1
                lines.append(line)
                line = ''
                char_flag = False

    tagger = ttw.TreeTagger(TAGLANG='en')

    word_lists = []
    for i in range(len(lines)):
        for t in tagger.TagText(lines[i]):
            surf = t.split()[0]        # 形態素
            base = t.split()[2]        # 基本形

            if re.match('V.*', t.split()[1]):
                pos = '動詞'
            elif re.match('WP.*', t.split()[1]) or re.match('PP.*', t.split()[1]):
                pos = '名詞'
            elif re.match('N.*', t.split()[1]):
                pos = '名詞'
            elif re.match('WRB', t.split()[1]) or re.match('EX', t.split()[1]) or re.match('R.*', t.split()[1]):
                pos = '副詞'
            elif re.match('J.*', t.split()[1]) or re.match('PDT', t.split()[1]):
                pos = '形容詞'
            else:
                pos = 'その他'

            word_lists.append([i, surf, base, pos])

    word_df = pd.DataFrame(word_lists, columns = ['セリフNo.', '単語', '基本形', '品詞'])
    word_df['品詞'] = word_df['品詞'].apply(lambda x : x.split(',')[0])

    score_result = pd.merge(word_df, posi_nega_df, on = ['基本形', '品詞'], how = 'left')

    result = []
    for i in range(len(score_result['セリフNo.'].unique())):
        temp_df = score_result[score_result['セリフNo.'] == i]
        text = ' '.join(list(temp_df['単語']))
        score = temp_df['スコア'].astype(float).sum()
        score_r = score / temp_df['スコア'].astype(float).count()
        result.append([i, text, score, score_r])

    result_df = pd.DataFrame(
        result,
        columns = ['セリフNo.', 'テキスト', '累計スコア', '標準化スコア']
    ).sort_values(by = 'セリフNo.').reset_index(drop = True)

    result_array = np.nan_to_num(result_df['標準化スコア'].values)

    return result_array




def run():
    import time
    import numpy as np
    import csv

    with open ('./title/title.txt', 'r', encoding='utf-8') as f:
        titles = f.read().split('\n')
    with open('./title/hero_name.txt', 'r', encoding='utf-8') as f:
        names = f.read().split('\n')
    with open('./title/score.txt', 'r', encoding='utf-8') as f:
        scores = f.read().split('\n')

    print('length = ', len(titles))
    for i in range(len(titles)):
        pn_data = []
        used_title = ''
        used_score = ''

        if names[i] == '':
            continue

        data = pn_score(title=titles[i], name=names[i])

        if np.sum(data) == 0:
            continue

        pn_data += data.tolist()
        used_title += titles[i] + '\n'
        used_score += scores[i] + '\n'

        time.sleep(2)

        if i == 0:
            mode = 'w'
        else:
            mode = 'a'
        with open('./data/pn_data.csv', mode=mode, encoding='utf-8') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(pn_data)
        with open('./data/title.txt',   mode=mode, encoding='utf-8') as f:
            f.write(used_title)
        with open('./data/score.txt',   mode=mode, encoding='utf-8') as f:
            f.write(used_score)

        print(i+1, ', ', end='', flush=True)


    print('Done!')
