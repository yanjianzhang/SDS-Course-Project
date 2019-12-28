import sys
import io
import os
input_stream = sys.stdin
# the map_input_file will be set by hadoop
# filepath = os.environ["map_input_file"]
# filename = os.path.split(filepath)[-1]
for line in input_stream:
    if line.strip()=="":
        continue
    # fields = line[:-1].split("\t")
    lineset = line[:-1].strip().split()
    # print(len(lineset))
    if len(lineset) == 2:
        location = lineset[1]
        idx = lineset[0]
        print(idx,"0",location)
        
    if len(lineset) > 4:
        date = lineset[0]
        time = lineset[1]
        filescale = lineset[2]
        project = lineset[3]
        idx = lineset[3][:2]
        print(idx,"1","\t".join((date,time,filescale,project)))
