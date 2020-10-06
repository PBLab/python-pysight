meta = """[MPA4A] 535
range=8128
periods=2
fstchan=0
holdafter=6e+006
sweepmode=35c8088
swpreset=65535
prena=4
syncout=8
fdac=0
rxdelay=0
cycles=2048
sequences=10
tagbits=15
dac0=21fc9
dac1=a037
dac2=a037
dac3=2037
dac4=2037
dac5=2037
rtpreset=7253.886
digio=600
digval=0
autoinc=1
savedata=3
mpafmt=asc
sephead=0
fmt=asc
smoothpts=5
[CHN1]
REPORT-FILE from 07/19/2017 06:15:58.539  written 07/19/2017 06:15:58
cmline0=07/19/2017 06:15:58.539
cmline1=A1
range=8128
active=1
bitshift=0
cftfak=2580100
evpreset=10
roimin=0
roimax=8100
caloff=0.000000
calfact=0.800000
calfact2=0
calfact3=0
calunit=nsec
caluse=1
[CHN2]
REPORT-FILE from 07/19/2017 06:15:58.539  written 07/19/2017 06:15:58
cmline0=07/19/2017 06:15:58.539
cmline1=A2
range=8128
active=1
bitshift=0
cftfak=2580100
evpreset=10
roimin=0
roimax=8100
caloff=0.000000
calfact=0.800000
calfact2=0
calfact3=0
calunit=nsec
caluse=1
[CHN3]
REPORT-FILE from 07/19/2017 06:15:58.539  written 07/19/2017 06:15:58
cmline0=07/19/2017 06:15:58.539
cmline1=A3
range=8128
active=1
bitshift=0
cftfak=2580100
evpreset=10
roimin=0
roimax=8100
caloff=0.000000
calfact=0.800000
calfact2=0
calfact3=0
calunit=nsec
caluse=1
;MPA4 #535
time_patch=5b
;datalength=8 bytes
;bit0..2: channel# 1..6 ( 3 bit)
;bit3: edge 0=up / 1=dn ( 1 bit)
;bit4 ..31: timedata    (28 bit), max sweep length 0.216 s
;bit32..47: sweeps      (16 bit)
;bit48..62: tag0..tag14 (15 bit)
;bit63:     data_lost   ( 1 bit)
[DATA]"""

meta_bytes = b"""[MPA4A] 535
range=8128
periods=2
fstchan=0
holdafter=6e+006
sweepmode=35c8088
swpreset=65535
prena=4
syncout=8
fdac=0
rxdelay=0
cycles=2048
sequences=10
tagbits=15
dac0=21fc9
dac1=a037
dac2=a037
dac3=2037
dac4=2037
dac5=2037
rtpreset=7253.886
digio=600
digval=0
autoinc=1
savedata=3
mpafmt=asc
sephead=0
fmt=asc
smoothpts=5
[CHN1]
REPORT-FILE from 07/19/2017 06:15:58.539  written 07/19/2017 06:15:58
cmline0=07/19/2017 06:15:58.539
cmline1=A1
range=8128
active=1
bitshift=0
cftfak=2580100
evpreset=10
roimin=0
roimax=8100
caloff=0.000000
calfact=0.800000
calfact2=0
calfact3=0
calunit=nsec
caluse=1
[CHN2]
REPORT-FILE from 07/19/2017 06:15:58.539  written 07/19/2017 06:15:58
cmline0=07/19/2017 06:15:58.539
cmline1=A2
range=8128
active=1
bitshift=0
cftfak=2580100
evpreset=10
roimin=0
roimax=8100
caloff=0.000000
calfact=0.800000
calfact2=0
calfact3=0
calunit=nsec
caluse=1
[CHN3]
REPORT-FILE from 07/19/2017 06:15:58.539  written 07/19/2017 06:15:58
cmline0=07/19/2017 06:15:58.539
cmline1=A3
range=8128
active=1
bitshift=0
cftfak=2580100
evpreset=10
roimin=0
roimax=8100
caloff=0.000000
calfact=0.800000
calfact2=0
calfact3=0
calunit=nsec
caluse=1
;MPA4 #535
time_patch=5b
;datalength=8 bytes
;bit0..2: channel# 1..6 ( 3 bit)
;bit3: edge 0=up / 1=dn ( 1 bit)
;bit4 ..31: timedata    (28 bit), max sweep length 0.216 s
;bit32..47: sweeps      (16 bit)
;bit48..62: tag0..tag14 (15 bit)
;bit63:     data_lost   ( 1 bit)
[DATA]"""
