#! /bin/awk -f
#
#
#        Jiggles a pdb file's coordinates by some random value                 -James Holton 3-17-24
#        run like this:
#
#        jigglepdb.awk -v seed=2343 -v shift=1.0 old.pdb >! jiggled.pdb
#         (use a different seed when you want a different output file)
#
#        to induce movements consistent with B factors, but with thru-bond correlated motions, do this:
#        jigglepdb.awk -v shift=byB -v seed=1234 \
#           -v frac_thrubond=0.5 -v ncyc_thrubond=100 \
#           -v frac_magnforce=5 -v ncyc_magnforce=50 \
#           -v shift_scale=1 -v dry_shift_scale=1 \
#           old.pdb >! jiggled.pdb
#
#        to just select alternate conformers and other partially-occupied atoms at random, but
#        only move xyz positions very slightly, do this:
#        jigglepdb.awk -v seed=$seed -v shift=byB -v shift_scale=0.01 \
#           -v independent_confsel=1 multiconf.pdb | grep -v " 0.00 " >! oneconf.pdb
#
BEGIN {

    if(! shift)  shift = 0.5
    if(! Bshift) Bshift = shift
    if(shift == "byB") Bshift = 0
    if(shift == "Lorentz") Bshift = 0
    if(! shift_scale) shift_scale = 1
    if(! dry_shift_scale) dry_shift_scale = 1
    pshift = shift
    shift_opt = shift
    if(pshift == "byB") pshift = "sqrt(B/8)/pi"
    if(pshift == "LorentzB") pshift = "Lorentzian B"
    if(seed) srand(seed+0)
    if(! keepocc) keepocc=0
    if(! independent_confsel) independent_confsel=0
    if(! disulfide_links) disulfide_links=0
    if(! distribution) distribution="gaussian";
    if(! Bavg_thrubond) Bavg_thrubond=0
    if(! frac_thrubond) frac_thrubond=0.5
    if(! ncyc_thrubond) ncyc_thrubond=0
    if(! frac_magnforce) frac_magnforce=5.0
    if(! ncyc_magnforce) ncyc_magnforce=ncyc_thrubond/2

    pi=4*atan2(1,1);

    # default to no global conformer selection
    global_confsel=-1;
    # doing independent selections for each conformer-residue
    if( independent_confsel ) {skipocc=1;keepocc=0};

    # automatically skip partial occupancies randomly if byB selected
    if(shift_opt=="byB" || shift_opt=="LorentzB" && ! skipocc) skipocc=1;

    print "REMARK jiggled by dXYZ=", pshift, "dB=", Bshift
    print "REMARK shift_scale=",shift_scale,"dry_shift_scale=",dry_shift_scale
    print "REMARK frac_thrubond=",frac_thrubond,"ncyc_thrubond=",ncyc_thrubond
    print "REMARK frac_magnforce=",frac_magnforce,"ncyc_magnforce=",ncyc_magnforce
    print "REMARK random number seed: " seed+0
    if(! keepocc && ! independent_confsel)
    {
      # random number between 1 and 0 to select all conformer choices
      global_confsel=rand();
      print "REMARK global conf sel: " global_confsel
    }
}

# count all lines
{++n;line[n]=$0}

/^ATOM|^HETAT/{
    if(debug) print tolower($0)

#######################################################################################
#    electrons = substr($0, 67,6)
#    XPLORSegid = substr($0, 73, 4)            # XPLOR-style segment ID
#    split(XPLORSegid, a)
#    XPLORSegid = a[1];
    Element = substr($0, 67)

#    Atomnum= substr($0,  7, 5)+0
    if(Element !~ /^[A-Z]/) Element= substr($0, 13, 2);
    Greek= substr($0, 15, 2);
    split(Element Greek, a)
    Atomtype[n]   = a[1];
    if(length(a[1])==4 && Element ~ /^H/)Element="H";
    gsub(" ","",Element);
    Ee[n] = Element;
    #main[n]=(Atom[n] ~ /^[NCO]$/ || Atom[n] ~ /^C[AB]$/);
    prefix[n] = substr($0,  1,30)
    Conf[n]   = substr($0, 17, 1)                # conformer letter
    Restyp[n] = substr($0, 18, 3)
    Segid[n]  = substr($0, 22, 1)            # O/Brookhaven-style segment ID
    Resnum[n] = substr($0, 23, 4)+0
    Insert[n] = substr($0, 27, 1)
    X[n]      = substr($0, 31, 8)+0
    Y[n]      = substr($0, 39, 8)+0
    Z[n]      = substr($0, 47, 8)+0
    Occ[n]    = substr($0, 55, 6)+0
    Bfac[n]   = substr($0, 61, 6)+0
    rest[n]   = substr($0, 67)
    # keep track of groupings
    atomtype = Atomtype[n]" "Restyp[n]
    residue = Restyp[n]" "Segid[n]" "Resnum[n] Insert[n];
    ++atoms_in[residue];
    sr = Segid[n]" "Resnum[n];
    csr = Conf[n]" "Segid[n]" "Resnum[n];

    if(Restyp[n]=="CYS") {
        ++has[sr];
        ++has[csr];
    }
#    ATOM   = toupper(substr($0, 1, 6))
#######################################################################################
}

# may want to "link" disulfide bonds
/^SSBOND/{
    ++found_disulfides

    c1=substr($0,15,1)
    s1=substr($0,16,1)
    r1=substr($0,18,4)+0
    c2=substr($0,29,1)
    s2=substr($0,30,1)
    r2=substr($0,32,4)+0

    sr1=s1" "r1
    sr2=s2" "r2
    csr1=c1" "s1" "r1
    csr2=c2" "s2" "r2

    disulfide_mate[sr2] = sr1
    disulfide_mate[sr1] = sr2
    disulfide_mate[csr2] = csr1
    disulfide_mate[csr1] = csr2
    print "REMARK recognized disulfide between:",csr1,csr2
}
/^LINK/{
    ++found_disulfides

    c1=substr($0,17,1)
    s1=substr($0,22,1)
    r1=substr($0,23,4)+0
    c2=substr($0,47,1)
    s2=substr($0,52,1)
    r2=substr($0,53,4)+0

    sr1=s1" "r1
    sr2=s2" "r2
    csr1=c1" "s1" "r1
    csr2=c2" "s2" "r2

    disulfide_mate[sr2] = sr1
    disulfide_mate[sr1] = sr2
    disulfide_mate[csr2] = csr1
    disulfide_mate[csr1] = csr2
    print "REMARK recognized disulfide between:",csr1,csr2
}

END{
  if(disulfide_links && ! found_disulfides) {
      print "REMARK warning, no disulfides found"
  }

  # min/max bond lengths
  if(min_bond_d == "") min_bond_d = 1
  if(max_bond_d == "") max_bond_d = 2

  # decide on shifts to apply
  for(i=1;i<=n;++i)
  {
    if(prefix[i] !~ /^ATOM|^HETAT/ ) continue;

    # abbreviations
    atomtype = Atomtype[i]" "Restyp[i]
    residue = Restyp[i]" "Segid[i]" "Resnum[i] Insert[i];
    sr = Segid[i]" "Resnum[i];
    csr = Conf[i]" "Segid[i]" "Resnum[i];

    # prepare to average over all atoms assigned to this conformer
    occsum[csr]+=Occ[i];
    ++count[csr];

    # sum over all atoms assigned to any conformer except default " "
    if(Conf[i]!=" ")
    {
      occsum[sr]+=Occ[i];
      ++count[sr];
    }

    if(shift_opt=="byB" || shift_opt=="LorentzB"){
      # switch on "thermal" shift magnitudes
      shift=sqrt(Bfac[i]/8)/pi*sqrt(3);

      # kick them more than byB?
      if(shift_scale != 1){
          shift *= shift_scale;
      }

      # kick them more if they are not water
      if(Restyp[i] != "HOH" && dry_shift_scale != 1){
          shift *= dry_shift_scale;
      }
    }

    if(skipocc)
    {
      # prepare to randomly "skip" conformers with occ<1
      # use first occurence to define occupancy of each conf letter
      if(Occ[i]+0<1){
        # only do this for first occurrence
        if(! seen[csr]){
          ++seen[csr];
          # record list of of partial-occ confs in each residue
          confletters[sr]=confletters[sr] Conf[i];
        }
      }
    }

    if(shift_opt == "LorentzB")
    {
        distribution = "Lorentz"
    }
    
    if(distribution == "gaussian" || distribution == "Gauss")
    {
        dx = gaussrand(shift/sqrt(3));
        dy = gaussrand(shift/sqrt(3));
        dz = gaussrand(shift/sqrt(3));
    }
    if(distribution == "uniform")
    {
        dR=2
        while(dR>1)
        {
            dx = (2*rand()-1);
            dy = (2*rand()-1);
            dz = (2*rand()-1);
            dR = sqrt(dx^2+dy^2+dz^2);
        }
        dx *= shift;
        dy *= shift;
        dz *= shift;
    }
    if(distribution == "Lorentz")
    {
        mag = lorentzrand(shift); 
        dR=2
        while(dR>1 || dR < 0.1)
        {
            dx = (2*rand()-1);
            dy = (2*rand()-1);
            dz = (2*rand()-1);
            dR = sqrt(dx^2+dy^2+dz^2);
        }
        dx *= mag/dR;
        dy *= mag/dR;
        dz *= mag/dR;
    }

    # now store the coordinate deltas
    dX[i] = dx;
    dY[i] = dy;
    dZ[i] = dz;

    # pick a random shift on B-factor or occ if selected
    if(Bshift+0>0) Bfac[i] += gaussrand(Bshift)
    if(Oshift+0>0) Occ[i] += gaussrand(Oshift)
  }

  # average atomic occupancies of each conf
  for(thing in occsum)
  {
    avgocc[thing]=occsum[thing]/count[thing];
    if(debug) print "REMARK DEBUG: average occ of",thing,"=",avgocc[thing]

  }

  # now that occupancies have been added up, fix a few things?
  for(sr in confletters)
  {
    # all these are residues with at least one atom with occ < 1

    # count number of conformer letters in this residue
    nconfs = length(confletters[sr]);
    if(debug) print "REMARK DEBUG:",sr,"has",nconfs,"conformer letters: |" confletters[sr] "|"

    # sometimes occupancies dont add up due to roundoff error
    roundoffsum=nconfs*sprintf("%.2f",1/nconfs);
    scale = 1./roundoffsum;
    sumocc = avgocc[sr]*nconfs;
    if(debug) print "REMARK DEBUG:",sr,"sum occ=",sumocc

    if( (sumocc >= 0.989 || sumocc <= 1.011 ) && nconfs > 3)
    {
         # this is probably a screwup
         roundoffsum=sumocc;
         scale = 1/roundoffsum;
    }

    if(sumocc == roundoffsum && scale != 1)
    {
      # high likelihood this is a round-off error
      csum=0;
      for(i=1;i<=nconfs;++i)
      {
        conf=substr(confletters[sr],i,1);
        csr = conf" "sr;
        # scale everything up to sum to unity
        avgocc[csr]*=scale;
        csum+=avgocc[csr];
      }
      print "REMARK WARNING: made occ sum=",sprintf("%.10g",sumocc)," scale to",csum,"for",sr;
    }

    if(global_confsel>=0)
    {
      # use same random number for all conformer choices
      # i.e. assume "A" is most popular and all "A" confs go together
      confsel_of[sr] = global_confsel;
    }

    # partial occupancies with single conformer should not use global confsel
    if(confletters[sr]==" " || independent_confsel){
      # select different confsel levels for each residue

      # being sure to connect disulfides
      if(confsel_of[sr]=="") {
        confsel_of[sr] = rand();
        if(disulfide_links && disulfide_mate[sr] != "") {
            confsel_of[disulfide_mate[sr]]=confsel_of[sr];
        }
        if(disulfide_links && disulfide_mate[csr] != "") {
            confsel_of[disulfide_mate[csr]]=confsel_of[csr];
        }
      }
    }

    # now go over conformer letters again
    cum_occ=0;
    for(i=1;i<=nconfs;++i)
    {
      conf=substr(confletters[sr],i,1);
      csr = conf" "sr;
      # define bracket of occupancy levels for selecting this conf
      lo=cum_occ;
      cum_occ+=avgocc[csr];
      hi=cum_occ;

      # conformer only exists if it falls in the chosen interval
      confsel = confsel_of[sr];
      if(lo < confsel && confsel <= hi){
        # lock in conformer letter
        confletter_of[sr]=conf;
        if(disulfide_mate[sr]!="" && has[conf" "disulfide_mate[sr]])
        {
            confletter_of[disulfide_mate[sr]]=confletter_of[sr];
        }
      }
      if(debug>2) print "REMARK DEBUG: confletter_of["sr"]= |"confletter_of[sr]"| confsel=",confsel_of[sr]
    }
  }

  # final pass to update occupancies of kept or rejected atoms  
  for(i=1;i<=n;++i)
  {
    if(! skipocc) continue;
    if(prefix[i] !~ /^ATOM|^HETAT/ ) continue;

    # abbreviations
    atomtype = Atomtype[i]" "Restyp[i]
    residue = Restyp[i]" "Segid[i]" "Resnum[i] Insert[i];
    sr = Segid[i]" "Resnum[i];
    csr = Conf[i]" "Segid[i]" "Resnum[i];

    if(Occ[i]+0<1) Occ[i] = 0;
    if(Conf[i] == confletter_of[sr]) Occ[i]=1;
    if(debug) print "REMARK DEBUG confsel",Conf[i],atomtype,residue,Occ[i]
  }


  if(frac_thrubond != 0 && ncyc_thrubond != 0)
  {
    print "REMARK measuring original bond lengths"
    # now, find all the bonds in the unperturbed structure that don't involve zero-occupancy members
    for(i=1;i<=n;++i){
      #skip zero occupancy
      if( Occ[i]==0 ) continue;
      for(j=1;j<i;++j){
        # skip big distances for speed
        if(X[i]>X[j]+3 || X[i]<X[j]-3) continue;
        if(Y[i]>Y[j]+3 || Y[i]<Y[j]-3) continue;
        if(Z[i]>Z[j]+3 || Z[i]<Z[j]-3) continue;
        # skip zero occupancy
        if( Occ[j]==0 ) continue;
        # skip nonsencial conformer relationships?
#           if( Conf[j] != Conf[i] && ! ( Conf[i] == " " || Conf[j] == " " || main[j] && main[i]) ) continue;
        mind=min_bond_d;maxd=max_bond_d;
        # hydrogen bonding lengths are shorter
        if( Ee[i] == "H" || Ee[j] == "H" ){
          mind=0.5;maxd=1.5;
          if(Ee[i]==Ee[j])maxd=0;
        };
        # recognize disulfides
        if( Ee[i]=="S" && Ee[j] == "S" && Restyp[i]=="CYS" && Restyp[j]=="CYS"){mind=1.5;maxd=2.5};
        # measure distance
        if(X[i]>X[j]+maxd || X[i]<X[j]-maxd) continue;
        if(Y[i]>Y[j]+maxd || Y[i]<Y[j]-maxd) continue;
        if(Z[i]>Z[j]+maxd || Z[i]<Z[j]-maxd) continue;
        d=sqrt((X[i]-X[j])^2+(Y[i]-Y[j])^2+(Z[i]-Z[j])^2);
        if(d>mind && d<maxd) {
          # distance falls within range
          newbond=1;
          for(k=1;k<=nbonds[i]+0;++k){
            # woops, this bond already exists
            if(bond[i,k]==j){
              newbond=0;
              break;
            }
          }
          if(newbond) {
            # legitimate new bond, increment the list
            ++nbonds[i];
            bond[i,nbonds[i]]=j;
            bondlen[i,j]=d;
          }
          newbond=1;
          # check if reverse bond is already there
          for(k=1;k<=nbonds[j]+0;++k){
            if(bond[j,k]==i){
              newbond=0;
              break;
            }
          }
          if(newbond) {
            # also account for reverse bond
            ++nbonds[j];
            bond[j,nbonds[j]]=i;
            bondlen[j,i]=d;
          }
        }
      }
    }
    print "REMARK done measuring bond lengths"

    # now maybe go through and force ideal bonds that we know about
#    for(i=1;i<=n;++i){
#      if(Occ[i]==0) continue;
#    }

    # measure and store all currently applied displacement magnitudes
    for(i=1;i<=n;++i){
      master_dXYZ[i]=sqrt(dX[i]**2+dY[i]**2+dZ[i]**2);
    }

    # iterative application of thru-bond smoothing
    for(l=1;l<=ncyc_thrubond;++l)
    {
      print "REMARK thru-bond averaging cycle",l
      # loop over all atoms
      for(i=1;i<=n;++i){
        if(Occ[i]==0) continue;
        realbonds=0;
        ddX[i]=ddY[i]=ddZ[i]=0;
        avgB[i]=Bfac[i];
        # take every atom bonded to this atom
        for(u=1;u<=nbonds[i];++u){
          j = bond[i,u];
          if(i==j) continue;
          if(Occ[j]==0)continue;
          ++realbonds;
          ddX[i] += dX[j];
          ddY[i] += dY[j];
          ddZ[i] += dZ[j];
          avgB[i] += Bfac[j];
        }
        if(! realbonds) continue;
        ddX[i] /= realbonds;
        ddY[i] /= realbonds;
        ddZ[i] /= realbonds;
        avgB[i] /= realbonds;
      }
      # second pass to update delta-positions
      for(i=1;i<=n;++i){
        if(Occ[i]==0) continue;
        if(nbonds[i]=="") continue;
        bw=frac_thrubond;
        if(Ee[i]=="H") bw=1;
        dX[i] = (1-bw)*dX[i] + bw*ddX[i];
        dY[i] = (1-bw)*dY[i] + bw*ddY[i];
        dZ[i] = (1-bw)*dZ[i] + bw*ddZ[i];
        bw=Bavg_thrubond;
        Bfac[i] = (1-bw)*Bfac[i] + bw*avgB[i];
      }
      # third pass to rescale shift magnitudes
      if(l<=ncyc_magnforce)
      {
        denom=ncyc_magnforce-1;
        if(denom==0)denom=1;
        mw=(1-(l-1)/denom)
        ms=frac_magnforce
        if(l>ncyc_magnforce)mw=0;
        for(i=1;i<=n;++i){
          if(Occ[i]==0) continue;
          if(nbonds[i]=="") continue;
          if(Ee[i]=="H") mw=0;
          mag = sqrt(dX[i]**2+dY[i]**2+dZ[i]**2);
          if(mag<=0.0) {
            continue;
            # make something up?
            mag=1e-6;
            dX[i]=mag*(rand()-0.5);
            dY[i]=mag*(rand()-0.5);
            dZ[i]=mag*(rand()-0.5);
          }
          # get difference between current and scaled version of originally prescribed shift magnitude
          dmag = (ms*master_dXYZ[i]-mag);
          # scale thet shift to be more like original magnitude
          scale = (mag+mw*dmag)/mag;
          dX[i] = scale*dX[i];
          dY[i] = scale*dY[i];
          dZ[i] = scale*dZ[i];
          if(i==debug) print "REMARK DEBUG",dX[i],dY[i],dZ[i],"    ",master_dXYZ[i],mag,"    ",dmag,scale,mw
        }
      }
    }
  }


  for(i=1;i<=n;++i)  
  {
    if(prefix[i] !~ /^ATOM|^HETAT/ )
    {
      print line[i];
      continue;
    }
  
    # apply the actual coordinate shift
    X[i] += dX[i];
    Y[i] += dY[i];
    Z[i] += dZ[i];
  
    # now print out the new atom
    printf("%s%8.3f%8.3f%8.3f %5.2f%6.2f%s\n",prefix[i],X[i],Y[i],Z[i],Occ[i],Bfac[i],rest[i]);        
  }
}



#######################################################################################
# function for producing a random number on a gaussian distribution
function gaussrand(sigma){
    if(! sigma) sigma=1
    rsq=0
    while((rsq >= 1)||(rsq == 0))
    {
        x=2.0*rand()-1.0
        y=2.0*rand()-1.0
        rsq=x*x+y*y
    }
    fac = sqrt(-2.0*log(rsq)/rsq);
    return sigma*x*fac
}

# function for producing a random number on a Lorentzian distribution
function lorentzrand(fwhm){
    if(! fwhm) fwhm=1

    return fwhm/2*tan(pi*(rand()-0.5))
}

function tan(x){
    return sin(x)/cos(x)
}
