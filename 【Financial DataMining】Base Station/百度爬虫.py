import http.client, urllib.request, urllib.parse, urllib.error, base64
from urllib.request import urlopen, quote
import pandas as pd
import json
from itertools import islice
import random
def dataloader():
    idx2loc = {}
    loc2idx = {}
    locs = []
    idxs = []
    f = open("station.csv")
    # 96+66+166
    for line in islice(f, 0, None):
        sp = line[:-1].split("\t")
        idx2loc[sp[0]] = (sp[1], sp[2])
        loc2idx[(sp[1], sp[2])] = sp[0]
        locs.append((sp[1], sp[2]))
        idxs.append(sp[0])
    f.close()
    return locs, idxs, loc2idx,idx2loc
locs, idxs, loc2idx,idx2loc = dataloader()
from urllib.parse import quote
import time
def getloc(loc):
    # 1
    samplex, sampley = loc
    url = 'http://api.map.baidu.com/geocoder'
    output = 'json'
    ak = 'eVHUnSglnzEgVxGuRq9rphmsrAWUbUzU' #
    coord_type = 'gcj02'
    uri = url + '?' + 'location=' + sampley + ','+samplex   + '&output=' + output + '&coord_type' + coord_type +  '&ak=' + ak  
    req = urlopen(uri)
        #     print(req)
    res = req.read().decode() 
#     print(res)
#     return None
    temp = json.loads(res)
    
    bus = temp['result']['business'].replace(",","-")
    firstout = bus +"&"+ temp['result']['formatted_address'].replace(",","-")
    # 2
    
    counts = []
    samplex, sampley = loc
    url = 'http://api.map.baidu.com/place/v2/search'
    output = 'json'
    ak = 'eVHUnSglnzEgVxGuRq9rphmsrAWUbUzU'
    coord_type = 'gcj02'
    selectList = ["写字楼 大厦","小吃 餐厅","商场 购物","学校 教育","景区","公寓 小区","博物馆 纪念馆" ,"工厂", "市场"]
    # 写字楼 
#     time.sleep(2)
    for item in selectList:
        # time.sleep(2)
        uri = url + '?' + 'query='+quote(item) + '&location=' + sampley + ','+samplex   + "&radius=500&" +  '&output=' + output + '&coord_type' + coord_type +  '&ak=' + ak  
        while True:
            try:
                    req = urlopen(uri)
                    res = req.read().decode() 
                    break
            except:
                    print("Error catched")
                    time.sleep(10)
                    continue
#         print(res)
        temp = json.loads(res)
        counts.append(len(temp['results']) if 'results' in temp else 0)
    print(counts)
    return [firstout] + counts
    

places = []
loc2place = {}
for index, loc in enumerate(locs):
#     print(loc)
    place = getloc(loc)
    places.append(place)
    loc2place[loc] = place
print(len(places))
fw = open('processedStation4.csv',"w")
for idx in idxs[:len(places)]:
    loc = idx2loc[idx]
    place = loc2place[loc]
    place = list(map(str,place))
    print(place)
    print(','.join(place))
    fw.write(idx + ',' + loc[0] + ',' + loc[1] + ','+ ','.join(place) + '\n')
fw.close()
    