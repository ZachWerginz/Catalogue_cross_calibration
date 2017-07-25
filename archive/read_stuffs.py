import t
import sys

print("Processing files...")

file_in = open('subaru.cutdown', 'r')
file_out = open("den.txt", "w")
file_err = open("badfiles.txt", "w")

c = 1
s = file_in.readline()
while s != '':
    try:
        dt = t.split_time(t.get_date_card(s[0:-1]))
        n = file_out.write(s[0:-1] + '\t' + dt + '\n')
    except IOError:
        file_err.write(s);

    s = file_in.readline()
    c += 1
    if (c % 100 == 0):
        file_out.flush()
        file_err.flush()
        print c,

file_err.close()
file_out.close()
file_in.close()

print("done");
