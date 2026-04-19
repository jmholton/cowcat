#! /bin/tcsh -f
#
#  set up occupancy refinement in refmac. Core documentation here:
#  http://www.ysbl.york.ac.uk/refmac/data/refmac_news.html#Occupancy
#
#
set pdbfile = "$1"
if(! -e "$pdbfile") then
    echo "usage: $0 refmacin.pdb"
    exit 9
endif
if("$2" =~ allhet) then
    goto allhet
endif
if("$2" =~ allatoms) then
    goto allatoms
endif
if("$2" =~ allres) then
    goto allres
endif

cat $pdbfile |\
awk '! /^ATOM|^HETAT/{next}\
  {segid=substr($0, 22, 1);resnum=substr($0, 23, 4);conf=substr($0, 17, 1);\
   occ=substr($0,55,6);atom=substr($0,13,4);Ee=$NF}\
  segid=="S" || /HOH|WAT/{conf="w"}\
  conf==" " && occ+0<1{print "_",segid,resnum,atom}\
  conf != " "{print conf,segid,resnum}' |\
sort -u | sort -k2,2 -k3g |\
awk '$1=="w"{$1=""} {print}' |\
awk 'NF==2{res=$1" "$2} NF>2{res=$2" "$3} \
     NF==4 && res==lastres{--n}\
     NF==4{print "occupancy group id",++n,"chain",$2,"residue",$3,"atom",$4}\
     NF==3{print "occupancy group id",++n,"chain",$2,"residue",$3,"alt",$1}\
     NF==2{print "occupancy group id",++n,"chain",$1,"residue",$2}\
     {lastres=res}' |\
awk 'BEGIN{print "occupancy refine"}\
     {print; res=$6" "$8;hasalt[res]=( / alt / )}\
    ! group[res]{++group[res];g[++n]=res}\
      ! ingroup[$4]{groups[res]=groups[res] " " $4;\
          ++count[g[n]];++ingroup[$4]} END{\
   for(i=1;i<=n;++i){\
      inc="";if(count[g[i]]==1 || ! hasalt[g[i]]) inc="in";\
      print "occupancy group alts " inc"complete", groups[g[i]];\
   }\
 }' |\
tee refmac_opts_occ.txt

exit

allhet:

cat $pdbfile |\
awk '! /^ATOM|^HETAT/{next}\
  {segid=substr($0, 22, 1);resnum=substr($0, 23, 4);\
   conf=substr($0, 17, 1);occ=substr($0,55,6)}\
   conf==" "{conf="n"}\
  {print conf,segid,resnum,( /^HETAT/ ),occ}' |\
awk '$1 != "n" || $4==1 || $5+0!=1' |\
awk '! seen[$1,$2,$3,$4]{++seen[$1,$2,$3,$4];print}' |\
awk '{++n;res[n]=$2" "$3;++count[res[n]];line[n]=$0} END{for(i=1;i<=n;++i){\
      print line[i],count[res[i]]}}' |\
awk 'BEGIN{print "occupancy refine"}\
     {conf=$1;chain=$2;resnum=$3;hetat=$4;occ=$5;res=$2" "$3;confs=$6}\
     {printf("occupancy group id %d residue %d ",++id,resnum)}\
     chain!=" "{printf("chain %s ",chain)}\
     conf !="n"{printf("alt %s",conf);}\
     {print ""}\
     confs>1 && ! hetat{groups[res]=groups[res]" "id;++taken[id]}\
  END{for(res in groups) print "occupancy group alts complete",groups[res];\
      for(i=1;i<=id;++i) if(! taken[i]) print "occupancy group alts incomplete",i}' |\
tee refmac_opts_occ.txt

exit


allatoms:
cat $pdbfile |\
awk '! /^ATOM|^HETAT/{next}\
     $NF=="H"{next}\
  {segid=substr($0, 22, 1);resnum=substr($0, 23, 4);\
   conf=substr($0, 17, 1);occ=substr($0,55,6);\
   atom=substr($0, 12, 5)}\
   segid==" "{segid="_"}\
   conf==" "{conf="_"}\
  {print conf,segid,resnum,atom,occ}' |\
awk 'BEGIN{print "occupancy refine"}\
     {conf=$1;chain=$2;resnum=$3;atom=$4;occ=$5;res=chain" "resnum;confs=$6}\
     {printf("occupancy group id %d residue %d ",++id,resnum)}\
     chain!=" "{printf("chain %s ",chain)}\
     conf !="_"{printf("alt %s",conf);}\
     atom !=""{printf("atom %s",atom);}\
     {print ""}\
  END{for(res in groups) print "occupancy group alts complete",groups[res];\
      for(i=1;i<=id;++i) if(! taken[i]) print "occupancy group alts incomplete",i}' |\
tee refmac_opts_occ.txt


exit


allres:
cat $pdbfile |\
awk '! /^ATOM|^HETAT/{next}\
     $NF=="H"{next}\
  {segid=substr($0, 22, 1);resnum=substr($0, 23, 4);\
   conf=substr($0, 17, 1);occ=substr($0,55,6);\
   atom=substr($0, 12, 5)}\
   segid==" "{segid="_"}\
   conf==" "{conf="_"}\
  {print conf,segid,resnum}' |\
awk '! seen[$0]{print;++seen[$0]}' |\
awk 'BEGIN{print "occupancy refine"}\
     {conf=$1;chain=$2;resnum=$3;atom=$4;occ=$5;res=chain" "resnum;confs=$6}\
     {printf("occupancy group id %d residue %d ",++id,resnum)}\
     chain!=" "{printf("chain %s ",chain)}\
     conf !="_"{printf("alt %s",conf);}\
     {print ""}\
  END{for(res in groups) print "occupancy group alts complete",groups[res];\
      for(i=1;i<=id;++i) if(! taken[i]) print "occupancy group alts incomplete",i}' |\
tee refmac_opts_occ.txt

