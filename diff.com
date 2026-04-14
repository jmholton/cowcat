#! /bin/tcsh -f
#
#        compute difference between first F and DANO in two MTZ files
#
#        -James Holton 5-23-19
#
set nawk = /bin/awk
$nawk 'BEGIN{print}' >& /dev/null
if($status) set nawk = awk
alias nawk $nawk
#
limit cputime 600
#
set tempfile = ${CCP4_SCR}/diff_temp$$
if($?DEBUG) set tempfile = diff_temp
mkdir -p ${CCP4_SCR}
#set logfile  = /dev/null
set logfile  = diff_details.log
set reffile = ""
set testfile = ""
set hiRES = ""
set loRES = ""

set isodiff = ""
set anodiff = ""

set weight = "NOWT"
set exclude = "EXCLUDE SIG 3"
set scale  = "refine anisotropic"
set ncyc = 4

# scan command line
set i = 0
while( $i < $#argv )
    @ i = ( $i + 1 )
    @ nexti = ( $i + 1 )
    @ lasti = ( $i - 1 )
    if($nexti > $#argv) set nexti = $#argv
    if($lasti < 1) set lasti = 1
    set arg = "$argv[$i]"

    if("$arg" =~ *=*) then
        set key = `echo $arg | awk -F "=" '{print $1}'`
        set val = `echo $arg | awk -F "=" '{print $2}'`
        if("$key" == "isodiff") set isodiff = "$val"
        if("$key" == "anodiff") set anodiff = "$val"
    endif

    # warn about probable mispellings
    if("$arg" =~ *.mtz) then
        if(-e "$arg") then
            if("$reffile" == "") then
                set reffile = "$arg"
            else
                set testfile = "$arg"
            endif
        else
            echo "WARNING: $arg does not exist"
        endif
        continue
    endif

    if("$arg" =~ "-"*) then
        # command-line option
        if("$arg" =~ "-wils"*) set scale = "refine wilson"
        if("$arg" =~ "-scale"*) set scale = "refine scale"
        if("$arg" =~ "-iso"*) set scale = "refine isotropic"
        if("$arg" =~ "-anis"*) set scale = "refine anisotropic"
        if("$arg" =~ "-nosca"*) set scale = "analyze"
        if("$arg" =~ "-nocyc"*) set ncyc = 0
        if("$arg" =~ "-excl"*) then
            set sig = `echo "$argv[$nexti]" | awk '{print $1+0}'`
            if("$sig" == "0" || "$sig" == "") set sig = 3
            set exlcude = "EXCLUDE SIG $sig"
            echo "excluding $sig -sigma reflections"
        endif
        if("$arg" =~ "-noex"*) set exclude = ""
        if("$arg" =~ "-weigh"*) set weight = "WEIGHT"
        if("$arg" =~ "-nowei"*) set weight = "NOWT"
        continue
    endif

    if("$arg" =~ [0-9]*) then
        # we have a number
        if(("$arg" =~ *A)||("$argv[$nexti]" == "A")) then
            # user-preferred resolution limits
            set temp = `echo "$arg" | nawk 'BEGIN{FS="-"} $1+0 > 0.1{print $1+0} $2+0 > 0.1{print $2+0}'`
            if($#temp != 1) then
                set temp = `echo $temp | nawk '$1>$2{print $1, $2} $2>$1{print $2, $1}'`
                if($#temp == 2) then
                    set loRES = "$temp[1]"
                    set hiRES = "$temp[2]"
                endif
            else
                if("$temp" != "") set hiRES = "$temp"
            endif
        endif
    endif
end

if( $#argv == 0 ) then
cat << EOF
usage: diff.com reference.mtz [Fref] test.mtz [Ftest]
where:
reference.mtz - is an mtz file containing the "right" structure factors
test.mtz      - is another mtz file containing structure factors to compare to it
Fref          - is the name of the column to use in reference.mtz (optional)
Ftest         - is the name of the column to use in test.mtz (optional)
2.0A          - only look at data out to 2.0 A resolution
2-6A          - only look at data in range 2A to 6 A resolution
-noscale      - no scaling, just difference
-wilson       - refine wilson scale
-scale        - scale factor only, no B
-iso          - refine scale and overall B factor
-aniso        - refine scale and anisotropic B factor
-nocyc        - do not refine
-weigh        - use sigmas as weights
-noweight     - no sigma weighting (default)
-exclude 3    - exclude reflections < 3x sigma
-noexclude    - do not exclude any reflections
EOF
    exit 9
endif

if(! -e "$reffile") then
    echo "ERROR: $reffile does not exist"
    exit 9
endif
if(! -e "$testfile") then
    echo "ERROR: $testfile does not exist"
    exit 9
endif

mtzdmp $reffile |\
awk '/OVERALL FILE STATISTICS/,/No. of reflections used/' |\
awk 'NF>10 && $(NF-1) ~ /^[A-Z]$/' |\
awk 'NF>2{print $(NF-1),$NF " "}' |\
cat >! ${tempfile}reflabels.txt
set Fref         = `awk '/^F/ && $2~/^F/{print $2;exit}' ${tempfile}reflabels.txt`
if("$Fref" == "") set Fref = `awk '/^F/{print $2;exit}' ${tempfile}reflabels.txt`
if("$Fref" == "") set Fref = `awk '/^G/{print $2;exit}' ${tempfile}reflabels.txt`
if("$Fref" == "") set Fref = `awk '/^R/{print $2;exit}' ${tempfile}reflabels.txt`
set DANOFref     = `awk '/^D/ && $2~/^DANO/{print $2;exit}' ${tempfile}reflabels.txt`
if("$DANOFref" == "") set DANOFref = `awk '/^D/{print $2;exit}' ${tempfile}reflabels.txt`
set SIGFref
set SIGDANOFref

mtzdmp $testfile |\
awk '/OVERALL FILE STATISTICS/,/No. of reflections used/' |\
awk 'NF>10 && $(NF-1) ~ /^[A-Z]$/' |\
awk 'NF>2{print $(NF-1),$NF " "}' |\
cat >! ${tempfile}testlabels.txt
set Ftest         = `awk '/^F/ && $2~/^F/{print $2;exit}' ${tempfile}testlabels.txt`
if("$Ftest" == "") set Ftest = `awk '/^F/{print $2;exit}' ${tempfile}testlabels.txt`
if("$Ftest" == "") set Ftest = `awk '/^G/{print $2;exit}' ${tempfile}testlabels.txt`
if("$Ftest" == "") set Ftest = `awk '/^R/{print $2;exit}' ${tempfile}testlabels.txt`
set DANOFtest     = `awk '/^D/ && $2~/^DANO/{print $2;exit}' ${tempfile}testlabels.txt`
if("$DANOFtest" == "") set DANOFtest = `awk '/^D/{print $2;exit}' ${tempfile}testlabels.txt`
set SIGFtest
set SIGDANOFtest



if("$hiRES" == "") then
    mtzdmp $testfile >! ${tempfile}mtzdump.txt
    set hiRES = `awk '/Resolution Range/{getline;getline;print $6}' ${tempfile}mtzdump.txt`
    rm -f ${tempfile}mtzdump.txt
endif


# run through args one more time to see if anything matches?
set i = 0
while( $i < $#argv )
    @ i = ( $i + 1 )
    @ nexti = ( $i + 1 )
    @ lasti = ( $i - 1 )
    if($nexti > $#argv) set nexti = $#argv
    if($lasti < 1) set lasti = 1
    set arg = "$argv[$i]"
    if("$arg" =~ *.mtz) continue

    set test = `awk -v arg="$arg" '$2==arg{print;exit}' ${tempfile}reflabels.txt`
    if( $#test == 2 && ! $?user_Fref) then
        if("$test[1]" == "Q" || "$test[1]" == "L" || "$test[1]" == "K") then
            set SIGFref = "$test[2]"
        else
            if("$test[1]" == "D") then
                set DANOFref = "$test[2]"
            else
                set Fref = "$test[2]"
                set user_Fref
            endif
        endif
        continue
    endif
    set test = `awk -v arg="$arg" '$2==arg{print;exit}' ${tempfile}testlabels.txt`
    if( $#test == 2) then
        if("$test[1]" == "Q" || "$test[1]" == "L" || "$test[1]" == "K") then
            set SIGFtest = "$test[2]"
        else
            if("$test[1]" == "D") then
                set DANOFtest = "$test[2]"
            else
                set Ftest = "$test[2]"
            endif
        endif
    endif
end

# sensible defaults for sigmas
if("$SIGFref" == "") set SIGFref      = `awk -v F="$Fref" '$NF==F{++f} f && /^[QLK] / && $NF~"^SIG"F{print $NF; exit}' ${tempfile}reflabels.txt`
if("$SIGFref" == "") set SIGFref = `awk -v F="$Fref" '$NF==F{++f} f && /^[QLK] /{print $NF; exit}' ${tempfile}reflabels.txt`
if("$SIGFref" == "") set SIGFref = `awk -v F="$Fref" '/^[QLK] /{print $NF; exit}' ${tempfile}reflabels.txt`
if("$SIGFtest" == "") set SIGFtest      = `awk -v F="$Ftest" '$NF==F{++f} f && /^[QLK] / && $NF~"^SIG"F{print $NF; exit}' ${tempfile}testlabels.txt`
if("$SIGFtest" == "") set SIGFtest = `awk -v F="$Ftest" '$NF==F{++f} f && /^[QLK] /{print $NF; exit}' ${tempfile}testlabels.txt`
if("$SIGFtest" == "") set SIGFtest = `awk -v F="$Ftest" '/^[QLK] /{print $NF; exit}' ${tempfile}testlabels.txt`

if("$SIGDANOFref" == "") set SIGDANOFref      = `awk -v DANOF="$DANOFref" '$NF==DANOF{++f} f && /^Q / && $NF~"^SIG"DANOF{print $NF; exit}' ${tempfile}reflabels.txt`
if("$SIGDANOFref" == "") set SIGDANOFref = `awk -v DANOF="$DANOFref" '$NF==DANOF{++f} f && /^Q /{print $NF; exit}' ${tempfile}reflabels.txt`
if("$SIGDANOFref" == "") set SIGDANOFref = `awk -v DANOF="$DANOFref" '/^Q /{print $NF; exit}' ${tempfile}reflabels.txt`
if("$SIGDANOFtest" == "") set SIGDANOFtest      = `awk -v DANOF="$DANOFtest" '$NF==DANOF{++f} f && /^Q / && $NF~"^SIG"DANOF{print $NF; exit}' ${tempfile}testlabels.txt`
if("$SIGDANOFtest" == "") set SIGDANOFtest = `awk -v DANOF="$DANOFtest" '$NF==DANOF{++f} f && /^Q /{print $NF; exit}' ${tempfile}testlabels.txt`
if("$SIGDANOFtest" == "") set SIGDANOFtest = `awk -v DANOF="$DANOFtest" '/^Q /{print $NF; exit}' ${tempfile}testlabels.txt`

rm -f ${tempfile}reflabels.txt ${tempfile}testlabels.txt


if("$Fref" == "") then
    set BAD = "no F in $reffile"
    goto cleanup
endif
if("$Ftest" == "") then
    set BAD = "no F in $testfile"
    goto cleanup
endif
if("$DANOFref" == "") set SIGDANOFref = ""
if("$DANOFtest" == "") set SIGDANOFtest = ""


echo -n "" >! $logfile

echo "F= $Fref SIGF= $SIGFref and DANO= $DANOFref SIGDANO= $SIGDANOFref in $reffile vs"
echo "F= $Ftest SIGF= $SIGFtest and DANO= $DANOFtest SIGDANO= $SIGDANOFtest in $testfile"
echo "$hiRES - $loRES A"


set cadfile     = cadded.mtz
set scaledfile  = scaleited.mtz
#
#
#################################################
# use CAD to put all data side-by-side
cad hklin1 $reffile hklout ${tempfile}Fref.mtz << EOF >> $logfile
labin file 1 E1=$Fref
ctypo file 1 E1=F
labou file 1 E1=Fref
xname file 1 all=ref
dname file 1 all=ref
EOF
cad hklin1 $testfile hklout ${tempfile}Ftest.mtz << EOF >> $logfile
labin file 1 E1=$Ftest
ctypo file 1 E1=F
labou file 1 E1=Ftest
xname file 1 all=test
dname file 1 all=test
EOF
if("$SIGFref" != "") then
    cad hklin1 $reffile hklout ${tempfile}SIGFref.mtz << EOF >> $logfile
labin file 1 E1=$SIGFref
ctypo file 1 E1=Q
labou file 1 E1=SIGFref
xname file 1 all=ref
dname file 1 all=ref
EOF
else
    # make one up
    set exclude = ""
    rm -f ${tempfile}SIGFref.mtz
    sftools << EOF >> $logfile
read ${tempfile}Fref.mtz
calc Q col SIGFref = col Fref 30 / ABS
write ${tempfile}SIGFref.mtz col SIGFref
quit
y
EOF
endif
if("$SIGFtest" != "") then
    cad hklin1 $testfile hklout ${tempfile}SIGFtest.mtz << EOF >> $logfile
labin file 1 E1=$SIGFtest
ctypo file 1 E1=Q
labou file 1 E1=SIGFtest
xname file 1 all=test
dname file 1 all=test
EOF
else
    # make one up
    set exclude = ""
    rm -f ${tempfile}SIGFtest.mtz
    sftools << EOF >> $logfile
read ${tempfile}Ftest.mtz
calc Q col SIGFtest = col Ftest 30 / ABS
write ${tempfile}SIGFtest.mtz col SIGFtest
quit
y
EOF
endif
if("$DANOFref" == "") then
    set NODANO
else
    cad hklin1 $reffile hklout ${tempfile}DANOFref.mtz << EOF >> $logfile
labin file 1 E1=$DANOFref
labou file 1 E1=DANOFref
xname file 1 all=ref
dname file 1 all=ref
EOF
    if("$SIGDANOFref" != "") then
        cad hklin1 $reffile hklout ${tempfile}SIGDANOFref.mtz << EOF >> $logfile
labin file 1 E1=$SIGDANOFref
labou file 1 E1=SIGDANOFref
xname file 1 all=ref
dname file 1 all=ref
EOF
    else
        # make one up
        set exclude = ""
        rm -f ${tempfile}SIGDANOFref.mtz
        sftools << EOF >> $logfile
read ${tempfile}DANOFref.mtz
calc Q col SIGDANOFref = col DANOFref 10 / ABS
write ${tempfile}SIGDANOFref.mtz col SIGDANOFref
quit
y
EOF
    endif
endif
if("$DANOFtest" == "") then
    set NODANO
else
    cad hklin1 $testfile hklout ${tempfile}DANOFtest.mtz << EOF >> $logfile
labin file 1 E1=$DANOFtest
labou file 1 E1=DANOFtest
xname file 1 all=test
dname file 1 all=test
EOF
    if("$SIGDANOFtest" != "") then
        cad hklin1 $testfile hklout ${tempfile}SIGDANOFtest.mtz << EOF >> $logfile
labin file 1 E1=$SIGDANOFtest
labou file 1 E1=SIGDANOFtest
xname file 1 all=test
dname file 1 all=test
EOF
    else
        # make one up
        set exclude = ""
        rm -f ${tempfile}SIGDANOFtest.mtz
        sftools << EOF >> $logfile
read ${tempfile}DANOFtest.mtz
calc Q col SIGDANOFtest = col DANOFtest 30 / ABS
write ${tempfile}SIGDANOFtest.mtz col SIGDANOFtest
quit
y
EOF
    endif
endif


# now cad them all together
cad hklin1 ${tempfile}Fref.mtz \
    hklin2 ${tempfile}SIGFref.mtz \
    hklin3 ${tempfile}Ftest.mtz \
    hklin4 ${tempfile}SIGFtest.mtz \
    hklout ${tempfile}Fs.mtz << EOF >> $logfile
RESOLUTION OVERALL $hiRES $loRES
labin file 1 all
labin file 2 all
labin file 3 all
labin file 4 all
xname file 2 all=ref
dname file 2 all=ref
xname file 3 all=test
dname file 3 all=test
xname file 4 all=test
dname file 4 all=test
EOF
if($status) then
    set BAD = "anomalous cad failed"
    goto cleanup
endif
rm -f ${tempfile}Fref.mtz ${tempfile}SIGFref.mtz
rm -f ${tempfile}Ftest.mtz ${tempfile}SIGFtest.mtz


if(! $?NODANO) then
    cad hklin1 ${tempfile}Fs.mtz \
        hklin2 ${tempfile}DANOFref.mtz \
        hklin3 ${tempfile}SIGDANOFref.mtz \
        hklin4 ${tempfile}DANOFtest.mtz \
        hklin5 ${tempfile}SIGDANOFtest.mtz \
        hklout ${tempfile}cad.mtz << EOF >> $logfile
RESOLUTION OVERALL $hiRES $loRES
labin file 1 all
labin file 2 all
labin file 3 all
labin file 4 all
labin file 5 all
xname file 2 all=ref
dname file 2 all=ref
xname file 3 all=ref
dname file 3 all=ref
xname file 4 all=test
dname file 4 all=test
xname file 5 all=test
dname file 5 all=test
EOF
    if($status) then
        set BAD = "anomalous cad failed"
        goto cleanup
    endif
else
    mv ${tempfile}Fs.mtz ${tempfile}cad.mtz
endif
rm -f ${tempfile}Fs.mtz
rm -f ${tempfile}DANOFref.mtz ${tempfile}SIGDANOFref.mtz
rm -f ${tempfile}DANOFtest.mtz ${tempfile}SIGDANOFtest.mtz


# test the ${tempfile}cad.mtz file to make sure it is good?

# the cad step was successful
cp -p ${tempfile}cad.mtz ${cadfile}


scaleit:
#################################################
set DANOstuff = "DPH1=DANOFtest SIGDPH1=SIGDANOFtest"
if($?NODANO) set DANOstuff = ""
set SIGFP = SIGFref
scaleit HKLIN ${tempfile}cad.mtz  HKLOUT ${tempfile}scaled.mtz << EOF-scaleit | tee ${tempfile}scaleit.log >> $logfile

TITLE Scale $testfile to ${reffile}.
RESO $hiRES $loRES        # Usually better to exclude lowest resolution data
#WEIGHT                  # Sigmas should be reliable.
#NOWT                    # ignore sigmas
#EXCLUDE SIG 3           # only use good spots for scaling
#refine isotropic        # use an isotropic B-factor
#refine anisotropic      # use an anisotropic B-factor
#analyze                # don't change relative scale
$weight
$exclude
$scale

LABIN FP=Fref  SIGFP=SIGFref -
    FPH1=Ftest SIGFPH1=SIGFtest $DANOstuff
CONV ABS 0.0001 TOLR 0.000000001 NCYC $ncyc
END
EOF-scaleit
if($status) then
    set BAD = "scaleit failed"
    goto cleanup
endif
# the scaleit step was successful
if( "$scale" == "analyze" ) then
   # scaleit doesnt do this anymore?
   cp ${tempfile}cad.mtz ${tempfile}scaled.mtz
endif
cp -p ${tempfile}scaled.mtz ${scaledfile}

set DANOstuff = "DANOFref SIGDANOFref"
if($?NODANO) set DANOstuff = ""
# generate a "perfectly scaled" copy of the input data (assuming Fref is on an absolute scale)
mtzutils hklin1 ${tempfile}scaled.mtz hklout absolute_scale.mtz << EOF-util >> $logfile
EXCLUDE Fref SIGFref $DANOstuff
END
EOF-util


egrep "Sc_kraut|TOTALS" ${tempfile}scaleit.log
set scale = `awk '$1=="Derivative" && ! /itle/{print $3}' ${tempfile}scaleit.log | tail -1`
set B     = `awk '/equivalent iso/{print $NF}' ${tempfile}scaleit.log | tail -1`
echo "scale= $scale B= $B"

if( "$isodiff" == "" ) then
    set isodiff = `awk '/acceptable differences/{gsub("*","x");print $NF}' ${tempfile}scaleit.log | head -1`
endif
echo "isomorphous difference cutoff: $isodiff"
if( "$anodiff" == "" ) then
    set anodiff = `awk '/acceptable differences/{gsub("*","x");print $NF}' ${tempfile}scaleit.log | tail -1`
endif
echo "anomalous difference cutoff: $anodiff"

# make a file with just Fs in it
cad \
   HKLIN1 ${tempfile}scaled.mtz \
HKLOUT ${tempfile}Fdiff.mtz << EOF-cad >> $logfile
LABIN  FILE 1 E1=SIGFtest E2=Ftest E3=Fref
END
EOF-cad

echo "     from $hiRES A to $loRES A"
echo "comparing $Fref in $reffile"
echo "       to $Ftest in $testfile"

# do some of our own statistics (assuming "ref" data are perfect
echo "RESO $loRES $hiRES\nNREF -1\nFORMAT '(3I4,3F30.20)'" |\
 mtzdump hklin ${tempfile}Fdiff.mtz |\
 awk '/ H K L /{for(i=4;i<=NF;++i){\
   if($i~/ref/)iref=i;\
   if($i~/test/)itest=i;\
   if($i~/SIG/)isig=i;};}\
   /LIST OF REFLECT/{++p} /Normal termination/{p=0}\
    p && NF==6 && ! /[a-z]/ && $6+0>0 && $4>-999{sig=$isig;Ftest=$itest;Fref=$iref;\
    print $1,$2,$3,Fref,Ftest,sig;}' |\
tee ${tempfile}Fdiff.txt |\
awk '{Fref=$4;Ftest=$5;sig=$6;print sqrt((Fref-Ftest)^2),sig,sqrt(Fref*Fref)}' |\
 awk -v isodiff=$isodiff '$1<=isodiff || $1<1.5*$2 || isodiff+0==0' |\
 awk '{sumdev+=$1;sumsig+=$2;sumcalc+=$3;++n}\
      $2+0>0{++m;sumdevsig+=$1/$2}\
 END{\
  nt=n;if(! n)n=1e-99;if(! m)m=1e-99;\
  dc=0;if(sumcalc+0!=0)dc=sumdev/sumcalc;\
  cd=0;if(sumdev+0!=0)cd=sumcalc/sumdev;\
  print "average Fref:",sumcalc/n,"n=",nt;\
  print "average deviation from correct F:",sumdev/n,dc*100"% F/s=",cd;\
  print "average sigma assigned to F:",sumsig/n;\
  print "average F X^2:",sumdevsig/m}'

cp -p ${tempfile}Fdiff.mtz Fdiff.mtz
cp -p ${tempfile}Fdiff.txt Fdiff.txt

if(! $?NODANO) then
    # make a file with ust anomalous differences in it
    cad \
   HKLIN1 $scaledfile \
HKLOUT ${tempfile}dano.mtz << EOF-cad >> $logfile
LABIN  FILE 1 E1=SIGDANOFtest E2=DANOFtest E3=DANOFref
#LABOUT FILE 1 ALLIN
END
EOF-cad

echo "RESO $loRES $hiRES\nNREF -1\nFORMAT '(3I4,3F30.20)'" |\
 mtzdump hklin ${tempfile}dano.mtz |\
 awk '/ H K L /{for(i=4;i<=NF;++i){\
   if($i~/ref/)iref=i;\
   if($i~/test/)itest=i;\
   if($i~/SIG/)isig=i;};}\
   /LIST OF REFLECT/{++p} /Normal termination/{p=0}\
     p && NF==6 && ! /[a-z]/ && $isig+0>0 && ! /999\.99999999/{sig=$isig;Dref=$iref;Dtest=$itest;\
     print $1,$2,$3,Dref,Dtest,sig}' |\
 tee ${tempfile}dano.txt |\
 awk '{Dref=$4;Dtest=$5;sig=$6;print sqrt((Dref-Dtest)^2),sig,sqrt(Dref*Dref),Dref,Dtest}' |\
 awk -v anodiff=$anodiff '$1<=anodiff' |\
 awk '{sumdev+=$1;sumsig+=$2;sumcalc+=$3;++n;\
    x=$4;y=$5;sumx+=x;sumy+=y;sumxy+=x*y;sumxx+=x*x;sumyy+=y*y;}\
      $2+0>0{++m;sumdevsig+=$1/$2}\
 END{\
  nt=n;if(! n)n=1e-99;if(! m)m=1e-99;\
  dc=0;if(sumcalc+0!=0)dc=sumdev/sumcalc;\
  cd=0;if(sumdev+0!=0)cd=sumcalc/sumdev;\
    avgx = sumx/n;avgy = sumy/n;avgxx = sumxx/n;avgxy = sumxy/n;avgyy = sumyy/n;\
    if(avgxx<avgx*avgx) avgxx=avgx*avgx;\
    if(avgyy<avgy*avgy) avgyy=avgy*avgy;\
    CC = 0;\
    if( (avgxy - avgx*avgy) == (sqrt(avgxx-avgx*avgx)*sqrt(avgyy-avgy*avgy)) ){\
        CC = 1;\
    }\
    if( avgxy == avgx*avgy) CC = 0;\
    if( avgxx>avgx*avgx && avgyy>avgy*avgy ) {\
       CC = (avgxy - avgx*avgy)/(sqrt(avgxx-avgx*avgx)*sqrt(avgyy-avgy*avgy))\
    }\
  print "average DANOFref:",sumcalc/n,"n=",nt;\
  print "average deviation from correct DANO:",sumdev/n,dc*100"% D/s=",cd,"CC=",CC;\
  print "average sigma assigned to DANO:",sumsig/n;\
  print "average DANO X^2:",sumdevsig/m}'

  cp -p ${tempfile}dano.mtz dano.mtz
  cp -p ${tempfile}dano.txt dano.txt
endif

cleanup:
# clean up?
if($?DEBUG) exit
rm -f ${tempfile}cad.mtz >& /dev/null
rm -f ${tempfile}Fref.mtz >& /dev/null
rm -f ${tempfile}Ftest.mtz >& /dev/null
rm -f ${tempfile}cadded.mtz >& /dev/null
rm -f ${tempfile}scaled.mtz >& /dev/null
rm -f ${tempfile}scaleit.log >& /dev/null
rm -f ${tempfile}Fdiff.mtz >& /dev/null
rm -f ${tempfile}Fdiff.txt >& /dev/null
rm -f ${tempfile}dano.mtz >& /dev/null
rm -f ${tempfile}dano.txt >& /dev/null

if(! $?BAD) then
    echo "Fs are now all in $cadfile"
    echo "DANOs are now all in dano.mtz"
else
    echo "$BAD"
    exit 9
endif

exit

###########
# other possibilities?

echo "labin file 1 E1=DANO E2=SIGDANO" | cad hklin1 merged.mtz hklout dano.mtz >& /dev/null
mtzdmp dano.mtz -1 | awk '{++count} $4*$4>9*$5*$5{++good} END{print good,count}'
set dano    = `echo "stats nbin 1 reso 1000 4" | mtzdump hklin $idealmtz | awk '$NF=="DANO"{print $8}'`
set sigdano = `echo "stats nbin 1 reso 1000 4" | mtzdump hklin scaleited.mtz | awk '$NF=="SIGDANO"{print $7}'`
echo "stats nbin 100 reso 4 2" | mtzdump hklin merged.mtz |\
 awk '$NF=="DANO" && ! /NONE/{res=$8;dano=$7;getline;print res,dano,$7}' |\
 awk '$3+0>0{print $1,$2/$3}' | tee dsig.txt |\
 tac |\
 awk '$2>1.0{thresh=$1;exit} END{print thresh+0"A is where DANO/SIGDANO drops below 1";exit}'

