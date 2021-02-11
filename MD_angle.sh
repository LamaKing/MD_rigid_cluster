#! /bin/bash

# Add executables from source folder
PATH="$PATH:$PWD"

# Create hexagonal cluster
# $2 is Bravais lat repetitions
# a lat vector
# b lat vector
cat << EOF > in_hex
$2 $2
4.45 0
-2.225  3.85381304684075
EOF

tool_create_hexagons.py in_hex > input_hex_5

# EP command line style
#echo 5 MD
#                      cluster       a   b   R    T    Fx    Fy    Nstep      dt
#MD_rigid_rototrasl.py input_hex_5   0.6 2.0 5.0  $T   $F    0     $N_steps   0.01

# AS json input       <input.json>
MD_rigid_rototrasl.py $1
