#! /bin/nawk -f
#
#
#	(re)build a molecule with a simple command-driven language
#
#
# commands are formatted like this:
# BUILD chain|resnum RES newatom oldatom1 oldatom2 oldatom3 chi angle dist
#
# EG:
# BUILD A21 ARG CA   N C CA -64 109.5 1.54
# BUILD A21 ARG CA   N C CA -64
#
#
BEGIN{

    # initialize "phantom" atoms to start phi-psi from
    N["X"]=0
    N["Y"]=0
    N["Z"]=0

    CA["X"]=-2.42
    CA["Y"]=0
    CA["Z"]=0

    C["X"]=-1.21
    C["Y"]=0
    C["Z"]=0.953

    O["X"]=0
    O["Y"]=0
    O["Z"]=0
}

/^CRYST/ {print}
/^REMARK/{print}
/^SCALE/ {print}

/^BUILD/ && NF >=6{
    # store building commands for later
    ++builds
    restyp = $2
    chain  = $3; if(chain !~ /^[A-Z]$/) chain = " "
    resnum = $4+0
    Build_typ[builds]    = restyp
    Build_num[builds]    = resnum
    Build_chain[builds]  = chain
    
    # reverse association
    BUILD[chain,resnum]  = builds
    # arbitrary number of phi-psi-chi angles
    for(i=5;i<=NF;++i) Build_angles[builds] = Build_angles[builds] $i
}

/^ATOM/ || /^HETATM/{
    # read any existing atoms into memory
    ++atoms;
    Original[atoms] = $0
    
    atomnum= substr($0,  7, 5)+0
    element= substr($0, 13, 2);
    greek= substr($0, 15, 2);
    split(element greek, a)
    atom   = a[1];
    chain  = substr($0, 22, 1)          # O/Brookhaven-style segment ID
    resnum = substr($0, 23, 4)+0
    restyp = substr($0, 18, 3)
    
    # count residues
    if(chain!=lastchain || resnum != lastresnum) ++residues
    
    # store all important items
    Atom[atoms]   = atom
    Restyp[atoms] = restyp
    Resnum[atoms] = resnum
    Chain[atoms]  = chain
    
    X[atoms]      = substr($0, 31, 8)+0
    Y[atoms]      = substr($0, 39, 8)+0
    Z[atoms]      = substr($0, 47, 8)+0
    Occ[atoms]    = substr($0, 55, 6)+0
    Bfac[atoms]   = substr($0, 61, 6)+0
    
    # set up associative memory
    atom_number[chain,resnum,atom] = atoms
    residue_number[chain,resnum] = residues
    
    # remember which atoms belong to which residue (for printing in order)
    Atoms_in[residues] = Atoms_in[residues] " " atoms
}



END{

    # now execute the (re)building commands
    for(build=1;build<=builds;++build)
    {
	# identify the indicated residue
	chain  = Build_chain[build]
	resnum = Build_num[build]
	restyp = Build_typ[build]
	
	# scan the atoms already built
	r = residue_number[chain,resnum]
	n=split(Atoms_in[i],a," ")
	for(j=1;j<=n;++j)
	{
	    num=a[j];
	    
	    if(Atom[num] == "CA")
	    {
		
	    }
	}
    }

    # now print the whole mess out
    for(r=1;r<=residues;++r)
    {
	n=split(Atoms_in[r],a," ")
	for(j=1;j<=n;++j)
	{
	    num=a[j];
	    ++atoms_printed;
	    
	    # skip all unmodified atoms (if desired)
	    if(onlynew && ! modified[num]) next
	    
	    printf("ATOM %6d  %-3s%1s%3s %1s%4d     %7.3f %7.3f %7.3f %5.2f%6.2f\n",\
atoms_printed,Atom[num],Conf[num],Restyp[num],Chain[num],Resnum[num],X[num],Y[num],Z[num],Occ[num],Bfac[num]);

	}
    }
}



################################################################################
#
#	build_ALA(N,CA,C)
#
#  	Function for building an alanine side chain
#
################################################################################
function build_ALA(N,CA,C) {
    # position of CB should be 120 degrees away?
    next_atom(C,N,CA,-120,110.5,1.52);
    CB["X"]=new_atom["X"]; CB["Y"]=new_atom["Y"]; CB["Z"]=new_atom["Z"];
    print_atom(CB,"CB");
    return 1
}


################################################################################
#
#	build_SER(N,CA,C,chi1)
#
#  	Function for building a serine side chain
#
################################################################################
function build_SER(N,CA,C,chi1) {
    # use alanine builder
    build_ALA(N,CA,C);

    # add oxygen with chi angle
    next_atom(N,CA,CB,chi1);
    OG["X"]=new_atom["X"]; OG["Y"]=new_atom["Y"]; OG["Z"]=new_atom["Z"];
    print_atom(OG,"OG");
    return 1
}


################################################################################
#
#	build_CYS(N,CA,C,chi1)
#
#  	Function for building a cystine side chain
#
################################################################################
function build_CYS(N,CA,C,chi1) {
    # use alanine builder
    build_ALA(N,CA,C);

    # add sulphur with chi angle
    next_atom(N,CA,CB,chi1);
    SG["X"]=new_atom["X"]; SG["Y"]=new_atom["Y"]; SG["Z"]=new_atom["Z"];
    print_atom(SG,"SG");
    return 1
}


################################################################################
#
#	build_THR(N,CA,C,chi1)
#
#  	Function for building a threonine side chain
#
################################################################################
function build_THR(N,CA,C,chi1) {
    # use alanine builder
    build_ALA(N,CA,C);

    # add CG and OG with chi angle
    next_atom(N,CA,CB,chi1);
    OG1["X"]=new_atom["X"]; OG1["Y"]=new_atom["Y"]; OG1["Z"]=new_atom["Z"];
    print_atom(OG1,"OG1");
    next_atom(N,CA,CB,chi1+120);
    CG2["X"]=new_atom["X"]; CG2["Y"]=new_atom["Y"]; CG2["Z"]=new_atom["Z"];
    print_atom(CG2,"CG2");

    return 1
}


################################################################################
#
#	build_VAL(N,CA,C,chi1)
#
#  	Function for building a valine side chain
#
################################################################################
function build_VAL(N,CA,C,chi1) {
    # use alanine builder
    build_ALA(N,CA,C);

    # add CGs with chi angle
    next_atom(N,CA,CB,chi1);
    CG1["X"]=new_atom["X"]; CG1["Y"]=new_atom["Y"]; CG1["Z"]=new_atom["Z"];
    print_atom(CG1,"CG1");
    next_atom(N,CA,CB,chi1+120);
    CG2["X"]=new_atom["X"]; CG2["Y"]=new_atom["Y"]; CG2["Z"]=new_atom["Z"];
    print_atom(CG2,"CG2");

    return 1
}


################################################################################
#
#	build_ILE(N,CA,C,chi1,chi2)
#
#  	Function for building a valine side chain
#
################################################################################
function build_ILE(N,CA,C,chi1,chi2) {
    # use valine builder
    build_VAL(N,CA,C,chi1);

    # add CD1 with 2nd chi angle
    next_atom(CA,CB,CG1,chi1);
    CD1["X"]=new_atom["X"]; CD1["Y"]=new_atom["Y"]; CD1["Z"]=new_atom["Z"];
    print_atom(CD1,"CD1");

    return 1
}


################################################################################
#
#	build_LEU(N,CA,C,chi1,chi2)
#
#  	Function for building a valine side chain
#
################################################################################
function build_LEU(N,CA,C,chi1,chi2) {
    # use alanine builder
    build_ALA(N,CA,C);

    # add CG with 1st chi angle
    next_atom(N,CA,CB,chi1);
    CG["X"]=new_atom["X"]; CG["Y"]=new_atom["Y"]; CG["Z"]=new_atom["Z"];
    print_atom(CG,"CG");

    # add CDs with 2nd chi angle
    next_atom(CA,CB,CG,chi2);
    CD1["X"]=new_atom["X"]; CD1["Y"]=new_atom["Y"]; CD1["Z"]=new_atom["Z"];
    print_atom(CG1,"CG1");
    next_atom(CA,CB,CG,chi2+120);
    CD2["X"]=new_atom["X"]; CD2["Y"]=new_atom["Y"]; CD2["Z"]=new_atom["Z"];
    print_atom(CG2,"CG2");

    return 1
}


# etc, etc


################################################################################
#
#	print_atom(atom, name, restyp, resnum, occ, Bfac)
#
#  	Function for printing a standard PDB line
#
################################################################################
function print_atom(atom, name, restyp, resnum, occ, Bfac) {
if(! name)   name = "CA"
if(! restyp) restyp = "ALA"
if(! resnum) resnum = 1
if(! occ)    occ = 1
if(! Bfac)   Bfac = 20
++atoms_printed

printf("ATOM %6d  %-3s %3s %1s%4d     %7.3f %7.3f %7.3f %5.2f%6.2f\n",\
atoms_printed,name,restyp," ",resnum,atom["X"],atom["Y"],atom["Z"],occ,Bfac);
}


################################################################################
#
#	next_atom(atom1,atom2,atom3, chi, angle, bond)
#
#  	Function for getting "new atom" xyz coordinates using:
#
#	three reference atoms (defining two "previous" bonds)
#	two   angles (the bond angle, and the chi torsion angle)
#	one   distance (length of the "new" bond)
#
#
#     O -atom1                     O - new_atom
#      \                          /
#       \                        /
#        \           angle      / - bond (1.54A)
#         \               \__  /
#          \   chi(= 0)   /   /
#    atom2- O -------------- O -atom3
#
# atom1, atom2, atom3, are 3-membered arrays ["X","Y","Z"]
# new atom returned in new_atom["X","Y","Z"]
#
# note: atom1 serves only to define chi=0, it can really be anywhere
################################################################################
#
#                                   |-> optional
function next_atom(atom1,atom2,atom3, chi, angle, bond) {


    # default bond angle and length of new bond
    if(chi   == "") chi   = 0
    if(angle == "") angle = 109.5
    if(bond == "")
    {
	bond = 1.54
	# assume some double-bond character in non-tetrahedral bonds
	if(angle != 109.5) bond = 1.4
    }
    

    # compute components of "new_atom"-"atom3" vector, relative to chi==0 defined by "bond1"
     axis_component["length"] = bond*sin(3.1415927*(angle-90)/180)
     chi0_component["length"] = -bond*cos(3.1415927*(angle-90)/180)*cos(3.1415927*(chi)/180)
    chi90_component["length"] = -bond*cos(3.1415927*(angle-90)/180)*sin(3.1415927*(chi)/180)

    # now we know the coordinates of the new atom vector in "local" coordinates
    # we need to construct a basis for converting them to "global" coordinates

    # vector subtration of atoms lying on rotation axis
    axis["X"] = atom3["X"]-atom2["X"]
    axis["Y"] = atom3["Y"]-atom2["Y"]
    axis["Z"] = atom3["Z"]-atom2["Z"]

    # vector subtraction of atoms defining "zero" rotation around the axis
    bond1["X"] = atom2["X"]-atom1["X"]
    bond1["Y"] = atom2["Y"]-atom1["Y"]
    bond1["Z"] = atom2["Z"]-atom1["Z"]

    # protect against singular vectors
    if(((axis["X"]^2 + axis["Y"]^2 + axis["Z"]^2) == 0)||((bond1["X"]^2 + bond1["Y"]^2 + bond1["Z"]^2)==0))
    {
	new_atom["X"] = 0;
	new_atom["Y"] = 0;
	new_atom["Z"] = 0; 
	return 0   
    }

    # normalize the "axis" vector
    axis["length"]  = sqrt( (axis["X"]*axis["X"])  + ( axis["Y"]*axis["Y"])  + (axis["Z"]*axis["Z"]));
    axis["X"] = axis["X"]/axis["length"]
    axis["Y"] = axis["Y"]/axis["length"]
    axis["Z"] = axis["Z"]/axis["length"]
    axis["length"] = 1

    # compute amount of "bond1" that needs to be removed in order to orthogonalize it to "axis"
    bond1_dot_axis  =     ( (axis["X"]*bond1["X"]) +  (axis["Y"]*bond1["Y"]) + (axis["Z"]*bond1["Z"]) ) # Dot product

    # subtract the projection from bond1
    bond1["X"] = bond1["X"] - bond1_dot_axis*axis["X"]
    bond1["Y"] = bond1["Y"] - bond1_dot_axis*axis["Y"]
    bond1["Z"] = bond1["Z"] - bond1_dot_axis*axis["Z"]

    # normalize the "bond1" vector to make the "chi0" reference vector
    bond1["length"] = sqrt((bond1["X"]*bond1["X"]) + (bond1["Y"]*bond1["Y"]) +(bond1["Z"]*bond1["Z"]) );
    chi0["X"] = bond1["X"]/bond1["length"]
    chi0["Y"] = bond1["Y"]/bond1["length"]
    chi0["Z"] = bond1["Z"]/bond1["length"]
    chi0["length"] = 1

    # now make the "other" basis vector to complete the right-handed, 
    # orthonormal basis, using a cross-product
    chi90["X"] = axis["Y"] * chi0["Z"] - axis["Z"] * chi0["Y"];
    chi90["Y"] = axis["Z"] * chi0["X"] - axis["X"] * chi0["Z"];
    chi90["Z"] = axis["X"] * chi0["Y"] - axis["Y"] * chi0["X"];
    
    # we now have three unit vectors forming a basis of rotation about the "atom2-atom3" bond

    # the "axis" component of the new atom, offset from atom3, will be 0.514A out along "axis"
    axis_component["X"]  = axis_component["length"]*axis["X"]
    axis_component["Y"]  = axis_component["length"]*axis["Y"]
    axis_component["Z"]  = axis_component["length"]*axis["Z"]

    # apply the "x-y" values from the given chi angle
    chi0_component["X"]  = chi0_component["length"]*chi0["X"]
    chi0_component["Y"]  = chi0_component["length"]*chi0["Y"]
    chi0_component["Z"]  = chi0_component["length"]*chi0["Z"]

    chi90_component["X"] = chi90_component["length"]*chi90["X"]
    chi90_component["Y"] = chi90_component["length"]*chi90["Y"]
    chi90_component["Z"] = chi90_component["length"]*chi90["Z"]

    
    # now generate a "new" atom, in the original coordinate system
    new_atom["X"] = atom3["X"] + axis_component["X"] + chi0_component["X"] + chi90_component["X"]
    new_atom["Y"] = atom3["Y"] + axis_component["Y"] + chi0_component["Y"] + chi90_component["Y"]
    new_atom["Z"] = atom3["Z"] + axis_component["Z"] + chi0_component["Z"] + chi90_component["Z"]

    return 1
}


function dihedral(atom1,atom2,atom3,atom4) {
#
#     O -atom1                     O - atom4
#      \                          /
#       \                        /
#        \                      /
#         \                    /
#          \      chi = 0     /
#    atom2- O -------------- O -atom3
#
# atom1, atom2, atom3, atom4, are 3-membered arrays ["X","Y","Z"]
# return value is chi
#
    
    # we need to construct a basis for converting them to "global" coordinates
    
    # get vector of first dihedral bond
    bond1["X"] = atom1["X"]-atom2["X"]
    bond1["Y"] = atom1["Y"]-atom2["Y"]
    bond1["Z"] = atom1["Z"]-atom2["Z"]
    
    # get vector of "second" (rotating axis) bond
    axis["X"]  = atom3["X"]-atom2["X"]
    axis["Y"]  = atom3["Y"]-atom2["Y"]
    axis["Z"]  = atom3["Z"]-atom2["Z"]

    # get vector of "third" dihedral bond
    bond3["X"] = atom4["X"]-atom3["X"]
    bond3["Y"] = atom4["Y"]-atom3["Y"]
    bond3["Z"] = atom4["Z"]-atom3["Z"]
    
    # normalize the "axis" to unit length
    norm = sqrt(axis["X"]^2 + axis["Y"]^2 + axis["Z"]^2)
    if(norm == 0) return "axis error"
    axis["X"] = axis["X"]/norm
    axis["Y"] = axis["Y"]/norm
    axis["Z"] = axis["Z"]/norm
    
    # reduce "bond" to their components perpendicular to the "axis"
    component  = bond1["X"]*axis["X"] + bond1["Y"]*axis["Y"] + bond1["Z"]*axis["Z"]
    bond1["X"] = bond1["X"]-component*axis["X"]
    bond1["Y"] = bond1["Y"]-component*axis["Y"]
    bond1["Z"] = bond1["Z"]-component*axis["Z"]
    
    component  = bond3["X"]*axis["X"] + bond3["Y"]*axis["Y"] + bond3["Z"]*axis["Z"]
    bond3["X"] = bond3["X"]-component*axis["X"]
    bond3["Y"] = bond3["Y"]-component*axis["Y"]
    bond3["Z"] = bond3["Z"]-component*axis["Z"]
    
    
    # now the angle between bond1 and bond3 is the dihedral angle
    

    # normalize the first and last bond vectors
    norm = sqrt(bond1["X"]^2 + bond1["Y"]^2 + bond1["Z"]^2)
    if(norm == 0) return "bond1 error"
    bond1["X"] = bond1["X"]/norm
    bond1["Y"] = bond1["Y"]/norm
    bond1["Z"] = bond1["Z"]/norm
    
    norm = sqrt(bond3["X"]^2 + bond3["Y"]^2 + bond3["Z"]^2)
    if(norm == 0) return "bond3 error"
    bond3["X"] = bond3["X"]/norm
    bond3["Y"] = bond3["Y"]/norm
    bond3["Z"] = bond3["Z"]/norm


    # construct a vector perpendicular to both the axis and bond1
    # (this differentiates "sides" of the dihedral)
    chi90["X"] = axis["Y"] * bond1["Z"] - axis["Z"] * bond1["Y"];
    chi90["Y"] = axis["Z"] * bond1["X"] - axis["X"] * bond1["Z"];
    chi90["Z"] = axis["X"] * bond1["Y"] - axis["Y"] * bond1["X"];
    
    
    # get the component of bond3 along bond1
    adjacent = bond1["X"]*bond3["X"] + bond1["Y"]*bond3["Y"] + bond1["Z"]*bond3["Z"]
    
    # get the component of bond3 along bond1
    opposite = chi90["X"]*bond3["X"] + chi90["Y"]*bond3["Y"] + chi90["Z"]*bond3["Z"]
    
    # use ArcTan to get the angle
    angle = atan2(opposite, adjacent)*180/3.1415927;
    
    return angle
}

