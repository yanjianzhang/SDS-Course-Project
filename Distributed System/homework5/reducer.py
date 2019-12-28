import sys
lastidx = ""
input_stream = sys.stdin
for line in input_stream:
    if line.strip() == "":
        continue
    lineset = line.strip().split()
    idx = lineset[0]
    if idx != lastidx:
        name = ""
        if lineset[1] == "0":
            name = lineset[2]
    elif idx == lastidx:
        if lineset[1] == "1":
            date = lineset[2]
            time = lineset[3]
            filescale = lineset[4]
            project = lineset[5]
            if name:
                print("\t".join((lastidx, name, date, time, filescale, project)))
    lastidx = idx
