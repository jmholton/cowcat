#! /bin/csh -f
#
#        Builds a PDB of evenly distributed, random atoms in a given
#        unit cell
#
#
set Vm = 2.4
set CELL = ( $1 $2 $3 $4 $5 $6 )
set SG = P1
set minD = 0
set N = ""

set i = 6
while ( $i < $#argv )
    @ i = ( $i + 1 )
    @ j = ( $i + 1 )
    set arg = "$argv[$i]"
    if("$arg" =~ [PpCcIiFfRrHh][1-6]*) then
        set temp = `echo $arg | awk '{print toupper($1)}'`
        set temp = `awk -v SG=$temp '$4 == SG {print $4}' $CLIBD/symop.lib | head -1`
        if("$temp" != "") then
            # add this SG to the space group list
            set SG = "$temp"
            continue
        endif
    endif
    if($j > $#argv) continue
    if("$arg" == "-minD") then
        set minD = "$argv[$j]"
        @ i = ( $i + 1 )
        continue
    endif
    if("$arg" == "-Vm") then
        set Vm = "$argv[$j]"
        @ i = ( $i + 1 )
        continue
    endif
    if("$arg" == "-N") then
        set N = "$argv[$j]"
        @ i = ( $i + 1 )
        continue
    endif
end

set SGnum = `awk -v SG=$SG '$4==SG {print $1}' ${CLIBD}/symop.lib | head -1`
set ASU_per_CELL = `awk -v SG=$SG '$4==SG {print $2}' ${CLIBD}/symop.lib | head -1`
set symops = `awk -v SG=$SG '$4==SG {print $2}' ${CLIBD}/symop.lib | head -1`

if($#CELL != 6) then
    cat << EOF
usage:

$0 50 60 70 90 90 90 P212121 -minD 1.5 -Vm

where the six numbers are the unit cell edge lengths and angles
P212121 is the desired space group
1.5 is the minimum distance between atoms
2.4 is the desired Matthews number

EOF
    exit 9
endif

# calculate maximum possible distance from origin
cat << EOF >! temp.xyz
    1   1.00000   1.00000   1.00000  20.00000 1.00    8         1OW1  WAT
    2   1.00000   1.00000   0.00000  20.00000 1.00    8         1OW1  WAT
    3   1.00000   0.00000   1.00000  20.00000 1.00    8         1OW1  WAT
    4   1.00000   0.00000   0.00000  20.00000 1.00    8         1OW1  WAT
    5   0.00000   1.00000   1.00000  20.00000 1.00    8         1OW1  WAT
    6   0.00000   1.00000   0.00000  20.00000 1.00    8         1OW1  WAT
    7   0.00000   0.00000   1.00000  20.00000 1.00    8         1OW1  WAT
    8   0.00000   0.00000   0.00000  20.00000 1.00    8         1OW1  WAT
EOF
coordconv XYZIN temp.xyz XYZOUT temp.pdb << EOF >> /dev/null
OUTPUT PDB ORTH 1
INPUT FRAC
CELL $CELL
END
EOF

set Xrange = `awk '{X=substr($0,31,8)+0} X<minX{minX=X} X>maxX{maxX=X} END{print minX+0,maxX+0}' temp.pdb`
set Yrange = `awk '{Y=substr($0,39,8)+0} Y<minY{minY=Y} Y>maxY{maxY=Y} END{print minY+0,maxY+0}' temp.pdb`
set Zrange = `awk '{Z=substr($0,47,8)+0} Z<minZ{minZ=Z} Z>maxZ{maxZ=Z} END{print minZ+0,maxZ+0}' temp.pdb`


if("$N" == "") then
    # estimate number needed to get this density
    set N = `echo $Xrange $Yrange $Zrange $Vm $ASU_per_CELL | awk '{print ($2-$1)*($4-$3)*($6-$5)/$7/14/$8}'`
endif

echo "generating an isotropic box of $N random atoms ..."
echo "$Xrange $Yrange $Zrange $N 1" |\
awk '{minX=$1;maxX=$2;minY=$3;maxY=$4;minZ=$5;maxZ=$6;N=$7*$8;srand();\
        for(i=1;i<=N;++i){\
            x=(maxX-minX)*rand()+minX;\
            y=(maxY-minY)*rand()+minY;\
            z=(maxZ-minZ)*rand()+minZ;\
            print x,y,z}\
     }' |\
awk '{printf"ATOM %6d  %-3s %3s %1s%4d     %7.3f %7.3f %7.3f %5.2f%6.2f\n",\
          1, "OW1", "WAT", " ", 1, $1, $2, $3, 1, 20}' |\
cat >! temp.pdb
echo "END" >> temp.pdb

echo "CELL $CELL" | pdbset xyzin temp.pdb xyzout raw.pdb >& /dev/null
echo "END" >> raw.pdb

# convert to fractional
echo "converting to fractional coordinates ..."
coordconv XYZIN raw.pdb XYZOUT temp.xyz << EOF >> /dev/null
INPUT PDB
OUTPUT FRAC
END
EOF

# cut out everything outside of cell boundaries
echo "cookie-cuttering out a unit cell ..."
cat temp.xyz |\
awk '$2>=0 && $2<1 && $3>=0 && $3<1 && $4>=0 && $4<1' |\
cat >! temp2.xyz
mv temp2.xyz temp.xyz

# convert back to PDB
echo "converting back to orthogonal space ... "
coordconv XYZIN temp.xyz XYZOUT temp.pdb << EOF >> /dev/null
OUTPUT PDB ORTH 1
INPUT FRAC
CELL $CELL
END
EOF

# now reformat the PDB
cat temp.pdb |\
awk '/^ATOM/{++i; printf"ATOM %6d  %-3s %3s %1s%4d     %7.3f %7.3f %7.3f %5.2f%6.2f\n",\
          i%100000, "OW1", "WAT", " ", i%10000, $6, $7, $8, 1, 20}' |\
cat >! random.pdb

mv random.pdb temp.pdb
pdbset xyzin temp.pdb xyzout random.pdb << EOF >& /dev/null
CELL $CELL
SPACE $SG
EOF

if("$minD" != "0") then

    echo "rejecting atoms that are < $minD from others in $SG "

    cat random.pdb >! probe_atoms.pdb
    set atoms = `egrep "^ATOM" probe_atoms.pdb | wc -l`
    set atom = 0
    while ( $atom < $atoms )
        @ atom = ( $atom + 1 )

        cat probe_atoms.pdb |\
        awk -v atom=$atom '/^ATOM/{++n} ! /^ATOM/ || n==atom{print}' |\
        cat >! probe_atom.pdb

        # see if it is still there
        cat probe_atom.pdb random.pdb | egrep "^ATOM" |\
        awk '{ID=substr($0,1,60);++seen[ID]} NR==1{ID0=ID}\
          END{print seen[ID0]}' >! test.txt
        set test = `cat test.txt`
        if($test == 1) then
            continue
        endif

        # gensym wont make more than 5000 atoms, so we have to do them one at a time
        gensym XYZIN probe_atom.pdb XYZOUT temp.pdb << EOF > /dev/null
        BROOK
        SYMM $SG
        XYZLIM -1.5 1.5 -1.5 1.5 -1.5 1.5
        READ probe_atom.pdb
EOF

        echo "ATOM BREAK $minD" |\
        cat temp.pdb - random.pdb |\
        awk '/^ATOM/' |\
        awk '/BREAK/{++b;minD=$3;next}\
                 {++n;X[n]=substr($0,31,8);Y[n]=substr($0,39,8);Z[n]=substr($0,47,8)}\
                 ! b{next}\
                b{\
                  for(i=1;i<n;++i){\
                    d=sqrt((X[n]-X[i])^2+(Y[n]-Y[i])^2+(Z[n]-Z[i])^2);\
                    if(d<=minD && d>0.001){next};\
                  }\
                print;}' |\
        cat >! new.pdb
        set test = `egrep "^ATOM" new.pdb | wc -l`
        echo "$test atoms left..."
        mv new.pdb random.pdb

    end

endif

cat random.pdb |\
awk -v N=$N '/^ATOM/{++n;if(n<=N)print}' |\
cat >! temp.pdb
pdbset xyzin temp.pdb xyzout random.pdb << EOF >& /dev/null
CELL $CELL
SPACE $SG
EOF

set atoms = `egrep "^ATOM" random.pdb | wc -l`
echo "$atoms atoms in random.pdb"

rm -f test.pdb
rm -f temp.pdb
rm -f raw.pdb
rm -f temp.xyz

exit


echo -n "" >! test.pdb
foreach da ( -1 1 0 )
foreach db ( -1 1 0 )
foreach dc ( -1 1 0 )
echo "TER $da $db $dc " >> test.pdb
echo "shift frac $da $db $dc" |\
  pdbset xyzin random.pdb xyzout temp.pdb > /dev/null
egrep ATOM temp.pdb >> test.pdb
end
end
end

egrep "^CRYST|^SCALE" random.pdb >! new.pdb
cat test.pdb |\
awk -v minD=$minD '/^TER 0 0 0/{++final} \
    /^ATOM/{++n;\
    X[n]=substr($0,31,8)+0;Y[n]=substr($0,39,8)+0;Z[n]=substr($0,47,8)+0;\
    if(final){\
        for(i=1;i<n;++i){\
            d=sqrt((X[i]-X[n])^2+(Y[i]-Y[n])^2+(Z[i]-Z[n])^2);\
            if(d<minD){\
                next;\
            }\
        }\
        print;\
    }\
} END{print "END"}' |\
cat >! new.pdb
mv new.pdb random.pdb

set atoms = `egrep "^ATOM" random.pdb | wc -l`
echo "$atoms left"

exit



# check the results

cat random.pdb |\
awk 'BEGIN{dmin=1e99} /^ATOM/{++n;\
  X[n]=substr($0,31,8)+0;\
  Y[n]=substr($0,39,8)+0;\
  Z[n]=substr($0,47,8)+0;\
  for(i=1;i<n;++i){\
    d=sqrt((X[i]-X[n])^2+(Y[i]-Y[n])^2+(Z[i]-Z[n])^2);\
    if(d<dmin){dmin=d;\
        print dmin;\
    };\
  }\
}'













echo "generating an isotropic box of $N random atoms ..."
echo "$Xrange $Yrange $Zrange $N" |\
awk '{minX=$1;maxX=$2;minY=$3;maxY=$4;minZ=$5;maxZ=$6;N=$7;srand();\
    Xrange=maxX-minX;Yrange=maxY-minY;Zrange=maxZ-minZ;\
    if(minD<=0)Xrange=Yrange=Zrange=0;\
    for(i=1;i<=N;++i){\
    dmin=-1;while(dmin<minD){\
        x=(maxX-minX)*rand()+minX;y=(maxY-minY)*rand()+minY;z=(maxZ-minZ)*rand()+minZ;\
        dmin=1e99;for(i=1;i<n;++i){\
            for(dX=-Xrange;dX<=Xrange;dX+=Xrange){\
            for(dY=-Yrange;dY<=Yrange;dY+=Yrange){\
            for(dZ=-Zrange;dZ<=Zrange;dZ+=Zrange){\
                d=sqrt((x-X[i]+dX)^2+(y-Y[i]+dY)^2+(z-Z[i]+dZ)^2);\
                if(d<dmin)dmin=d;if(d<minD)break;\
            }}}\
        };\
    }\
    ++n;X[n]=x;Y[n]=y;Z[n]=z;\
    print x,y,z}}' |\
awk '{printf"ATOM %6d  %-3s %3s %1s%4d     %7.3f %7.3f %7.3f %5.2f%6.2f\n",\
          1, "OW1", "WAT", " ", 1, $1, $2, $3, 1, 20}' |\
cat >! temp.pdb
echo "END" >> temp.pdb
