#-*- encoding:UTF-8 -*- 

import re

sec_start = 60*37+24
tmp_time = 0

for line in open("/Users/ikonka/PycharmProjects/6D_final/data/Needlelogs_2.txt"):
    words = re.split(r"x:|,y:|z:|\(|\)|, |;",line.strip())
    time = words[0]
    times = re.split(r":| ", time)
    min = int(times[3])
    sec = int(times[4])
    sec_time = min*60+sec
    if (sec_time != tmp_time) and ((sec_time-sec_start) >= 0):
        tmp_time = sec_time
        num = 30*(sec_time-sec_start)
        trans_matrix = [words[2], words[3], words[5]]
        rotation_matrix = words[12:16]
        out_file = open('data/labels_pose/2_%d.txt'%num, 'w')
        out_file.write( " ".join([a for a in trans_matrix])+" "+" ".join([b for b in rotation_matrix]))

    