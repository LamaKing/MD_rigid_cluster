N=7
cat << EOF > input.hex
$N $N 
4.45   0.0
-2.225 3.853813046840752 
EOF

# Static maps
./static_roto_map.py input-static.json $N $((N+1)) 

# MD
./tool_create_cluster.py input.hex hexagon > input_hex-Nl_7.hex # Same file as input.json
./MD_rigid_rototrasl.py input-MD.json > out-MD.dat
