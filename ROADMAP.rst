Initial code idea:

y0 = [1.0, 0.0]
freeenergies = [0.0, 20.0, -10.0]

model = parse("A -> A‡ -> B")
# model.compounds
# model.reactions
# model.A
# model.is_half_equilibrium

k = eyring(model.B @ freeenergies)

dydt = get_dydt(model, k)
y, r = get_y(dydt, y0)
y(y.t_max)

In some advanced tutorial, I might show how to create a polymerization model
(Pn + P -> Pn+1‡ -> Pn+1) using data extrapolated from data on small to
medium n.

In the future, I will allow an interface to ASE output files as well. That,
together with AMP, might turn into a successful post-doc.

Please take a look at our :doc:`tutorials/index`!

Developers
----------

I use flake8 to ensure style. The following plugins are used:
- pyflakes
- flake8-bugbear
- flake8-docstrings
- pycodestyle
- pydocstyle
- hacking.core
- mccabe

I use black to ensure formatting. flake8 configuration is such that it is
minimally compatible with black.

Extras #1
---------

The thing is to go over the normal workflow from the last to the start,
writing the required functions with one example and a test for each major use
case.
If some thing goes wrong we'll know. Tests will be based on reference values
available from the literature or other codes (it is nice if we have a
citation here).

- I believe the slowest rate constant of an equilibrium should probably be 10
  to 1000 times faster than the fastest reaction constant, but this might
  depend on a case-by-case basis, so the user should have some freedom here

The future CLI
--------------

THE MASTER IDEA IS THAT YOU COMPILE A MODEL AND THEN USE IT FOR ANALYZING.

$ ls
model.k
E.out
S.out
ES.out
TS.out
P.out
$ cat model.k
# input file for the model
...
$ ovrct --compile model.k
$ ls
model.k
model.rct
...

Things I want to incorporate:
- The above is actually a simple code that a. receives reaction scheme from an
  input file and b. reads a list of energies and other data per compound
  (processed from logfiles) and returns a list of reaction rate constants.
  This is basically the core of the compiled model. NEVER FORGET: THIS IS
  YOUR GREATER INNOVATION. This will allow e.g. the study of concurrent
  reactions and selectivity (final ratio between possible products,
  percentage of desired product produced compared with that could be
  maximally achieved), among other things.
- Analyses should ideally receive only the reaction scheme and the simulation
  results (t and y); alternatively, dydt and the reaction scheme. Some other
  analyses might require the original absolute free energies (e.g., stability
  analysis, degree of rate control), while others might need to recalculate
  everything from scratch (Arrhenius plots, isotope effects). All the needed
  data should be included in the architected compiled file format.
- When further processing the compiled model, it may be important to recreate
  a representation of the reactions and equilibria. This is accomplished with,
  among others, is_half_quilibrium, allowing _unparse_reaction to write the
  original equilibria (the hardest part to get back intact)
- With all this, it might be wise to allow recompilation from a previous
  compiled model, even allowing to create alternative models with some small
  differences among them.

What GoodVibes has and I want:
A Python program to compute quasi-harmonic thermochemical data from any
frequency calculation logfile at a given temperature/concentration, corrected
for the effects of vibrational scaling-factors and available free space in
solvent. All (electronic, translational, rotational and vibrational) partition
functions are recomputed and will be adjusted to any temperature or
concentration. These default to 298.15 K and 1 atmosphere.

Future input file
-----------------

It might be useful to allow charges in compound names, which will demand we
require spaces around "+" when used to separate reactants and products.

I don't think cclib is 100% ready yet and, as such, and because I don't want
to reinvent the wheel and parse everything, I decided to design an input file
that includes everything, but is also extendable to include paths to logfiles
in the future.

    $scheme  // I know how to parse this
     // REACTION: CH4 + Cl ---> CH3 + HCl
     CH4 + ·Cl   -> [H3C·H·Cl]‡
     [H3C·H·Cl]‡ -> ·CH3 + HCl
    $end

    // Rules for parsing options:
    // 1. key=value
    // 2. key is string
    // 3. if value has comman, it is a sequence
    // 4. if value or value members look like numbers, they are numbers
    //    if value or value members are true or false, they are booleans
    //    otherwise, they are strings
    $options
     username=violeta
     date=2018-09-12 12:29:03
     method=tst  // default
     tunnel=eck  // default
     // If I ever implement some way of recalculating the energies in every
     // temperature, this will be the input for it:
     // temperature=200, 298.15, 300, 400  // temperatures in kelvin!
    $end

    // All compound below will be read and the analysis will be made for all of
    // common temperatures in the logfiles. Logfiles are check for having the
    // same level of theory if possible (here UMP2/6-311G(3d,2p)).
    $compounds
     [H3C·H·Cl]‡:
       logfile=ch4cl_ts_mp2_3d2p.out
       freeenergy=...
       scfenergy=...
       nsym=3
       // rxsym=4  // this is not needed if we add all nsym
     CH4:
       logfile=ch4_mp2_3d2p.out
       freeenergy=...
       scfenergy=...
       nsym=12  // alternatively, we could receive the name of the point group
     ·CH3:
       logfile=ch3_mp2_3d2p.out
       freeenergy=...
       scfenergy=...
       nsym=1
     HCl:
       logfile=hcl_mp2_3d2p.out
       freeenergy=...
       scfenergy=...
       nsym=1
     ·Cl:
       logfile=cl_mp2_3d2p.out
       freeenergy=...
       scfenergy=...
       nsym=6
    $end
    // EOF

    $scheme
     NH3 + ·OH -> [NH3OH]‡ -> ·NH2 + H2O
    $compounds
     NH3:
       logfile=nh3_m062x.out
       radius=2.59
     ·OH:
       logfile=oh_m062x.out
       radius=2.71
     [NH3OH]‡:
       logfile=nh3oh_ts_m062x.out
       rxsym=3
       rxd=2.6
     ·NH2:
       logfile=nh2_m062x.out
     H2O:
       logfile=h2o_m062x.out
    $options
     method=tst
     tunnel=eck
     diff=true
     ab=0
    $end
    // EOF

    $scheme
     E + S <=> ES
     ES -> P + E
    $compounds
     E: ...
     ES: ...
     S: ...
     P: ...
    $end

What we expect and which are the defaults
-----------------------------------------

r#, p#, rc, pc, ts <- frequency logfile, [single point logfile] (rc, pc and ts stand for reaction and product complexes and transition state, respectively)
nsym_r#, nsym_p#, etc <- rotational symmetry number for the structures above (overrides the one in the logfile)
rxsym <- degeneracy path or reaction symmetry (possible ways of reacting)
method <- currently, only "tst" (default)
tunnel <- either "wigner" or "eckart" (default)
temp <- list of temperatures, most common temperature in logfiles by default (if not all the same, a warning is given)
diffusion <- True (default) or False (whether to include diffusion effects when in solution)
visc <- viscosity, if no solvent is given, at every temperature

I don't know what to do with PRODVn, AB, ET, RXD, RADn, PH, SPH and PKA_R1

uni-, bimolecular (also pre-reactive complex)
rate constants using transition state theory
canonical emsemble
wigner, eckart
collins-kimball for diffusion limit
Marcus theory for electron transfer
molar fractions for pH

Things I which cclib could read from ORCA logfiles
--------------------------------------------------
- Absolute free and electronic energies

Approximations per paper
------------------------
Items with an * are not necessary in our present approach, or are
incorporated in chunks compatible with our methodology, but the effects are
still taken in consideration.

DOI:10.1002/qua.25686:
- Corrections for reactions in solution:
  - Diffusion effect through Collins-Kimball theory
  - Electron transfer through Marcus theory
  - Some of the above are from QM-ORSA for reactions in solution*

DOI:10.1039/C5CP00628G:
- Anharmonicity
- Molecular charge

DOI:10.1021/acs.jpca.8b06092:
- Variational transition state theory
- Small curvature tunneling

DOI:10.1021/acscatal.7b00115:
- Degree of rate control
- Degree of selective control
- Brønsted-Evans-Polanyi (BEP) relations
- Use of degree of rate control under transient reaction conditions
- Use of degree of rate control to choose computational models at low level
- Use of degree of rate control for screening catalysts

DOI:10.1039/c8cs00398j (lots of interesting things, some highlighted below):
- Apparent activation energy
- Degree of rate control
- Linear free energy relationships
- Process optimisation

DOI:10.1002/cphc.201100137:
- Slowest step of the reaction
- Step with smallest rate constant
- Step with highest free energy transition state
- Step with rate constant that exerts the strongest effect
- Energetic span model

DOI:10.1002/anie.200462544:
- Reaction progress kinetic analysis
- Differential and integral measurements
- Data interrogation
- Catalyst induction periods
- Catalyst deactivation and product inhibition
- Catalyst resting states
- Reaction order and turnover frequency

Things to do after going public
-------------------------------

- Describe each submodule in the docs
- Rebase to a single commit?
- Publish article (with some guidelines on how to properly calculate good
equilibria and, consequently, reaction rate constants)

These are the approximations available from the KiSThelP (<http://kisthelp.univ-reims.fr/userDocumentation/calculationMenu.html>):
- Gas phase chemical equilibrium constants
- Transition state theory
- Transition state theory with Wigner tunneling
- Transition state theory with Eckart tunneling
- Variational transition state theory
- Variational transition state theory with Wigner tunneling
- Variational transition state theory with Eckart tunneling
- Rice-Ramsperger-Kassel-Marcus (RRKM)

These are the approximations available from MKMCXX (<https://wiki.mkmcxx.nl/index.php/Main_Page>):
- Except for the GUI, the program works reasonably nice in terms of output files written and input style, I like that
- Calculation of reaction orders
- Calculation of apparent activation energies
- Degree of rate control analysis
- Thermodynamic degree of rate control analysis
- Degree of selectivity control analysis
- Multiplier used to speed-up reaction rates (booster, sometimes leads to faster convergence towards the steady-state solution)
- Turn-over-frequencies as a function of temperature
- Selectivity between products as a function of temperature
- Degree of selectivity coefficient for a particular product as a function of temperature
- Degree of selectivity control heatmap
- Surface coverage as a function of time
- Final surface coverage for adsorbant compounds as a function of temperature

Use-case stories
----------------

- I calculated a reaction scheme and I want to know the rate determining step
- I calculated a reaction scheme and I want to know the final proportion of products
- I want to know the kinetic isotope effect for a calculated reaction scheme
- I want to know the apparent activation energy for product formation
- I want to know the apparent order on a particular reactant
