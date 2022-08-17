#!/bin/bash
# script for variing choosen parameter of the classifiers 

cd ~/Documents/SMDA2/SMDA2/termPaper/projectCode

if ! [ -d "varOut" ]
then
    echo "creating varOut"
    mkdir varOut
fi

cuts="0 -2 -5"
dcut="0"

layers="4 8 12"
dlayer="12"

baseLr="1e-3 1e-4 1e-2"
dbaseLr="1e-3"
endLr="1e-4 1e-5 1e-3"
dendLr="1e-4"

hidden_width="200 300 10"
dhidden_width="200"
hidden_depth="1 2 3"
dhidden_depth="2"

n_hidden_w="100 10 200"
dn_hidden_w="100"
n_hidden_d="1 2 4"
dn_hidden_d="2"

batchsize0="100 1000 2000"
dbatchsize0="1000"
batchsize1="100 10 200"
dbatchsize1="100"

# python3 classifieranalysis.py dbatchsize0 dbatchsize1 dcut dhidden_width dhidden_depth dn_hidden_w dn_hidden_d dbaseLr dendLr dlayer

python3 classifieranalysis.py $dbatchsize0 $dbatchsize1 $dcut $dhidden_width $dhidden_depth $dn_hidden_w $dn_hidden_d $dbaseLr $dendLr $dlayer

for cut in $cuts
do 
    python3 classifieranalysis.py $dbatchsize0 $dbatchsize1 $cut $dhidden_width $dhidden_depth $dn_hidden_w $dn_hidden_d $dbaseLr $dendLr $dlayer
done

for layer in $layers
do
    python3 classifieranalysis.py $dbatchsize0 $dbatchsize1 $dcut $dhidden_width $dhidden_depth $dn_hidden_w $dn_hidden_d $dbaseLr $dendLr $layer
done

for b0 in $batchsize0
do
    python3 classifieranalysis.py $b0 $dbatchsize1 $dcut $dhidden_width $dhidden_depth $dn_hidden_w $dn_hidden_d $dbaseLr $dendLr $dlayer
done

for b1 in $batchsize1
do
    python3 classifieranalysis.py $dbatchsize0 $b1 $dcut $dhidden_width $dhidden_depth $dn_hidden_w $dn_hidden_d $dbaseLr $dendLr $dlayer
done


# for dhw in $hidden_width
# do 
#     python3 classifieranalysis.py $dbatchsize0 $dbatchsize1 $dcut $dhw $dhidden_depth $dn_hidden_w $dn_hidden_d $dbaseLr $dendLr $dlayer
# done

# for dhd in $hidden_depth
# do 
#     python3 classifieranalysis.py $dbatchsize0 $dbatchsize1 $dcut $dhidden_width $dhd $dn_hidden_w $dn_hidden_d $dbaseLr $dendLr $dlayer
# done

for nhw in $n_hidden_w
do 
    python3 classifieranalysis.py $dbatchsize0 $dbatchsize1 $dcut $dhidden_width $dhidden_depth $nhw $dn_hidden_d $dbaseLr $dendLr $dlayer
done

for nhd in $n_hidden_d
do 
    python3 classifieranalysis.py $dbatchsize0 $dbatchsize1 $dcut $dhidden_width $dhidden_depth $dn_hidden_w $nhd $dbaseLr $dendLr $dlayer
done


for blr in $baseLr
do 
    python3 classifieranalysis.py $dbatchsize0 $dbatchsize1 $dcut $dhidden_width $dhidden_depth $dn_hidden_w $dn_hidden_d $blr $dendLr $dlayer
done

for elr in $endLr
do 
    python3 classifieranalysis.py $dbatchsize0 $dbatchsize1 $dcut $dhidden_width $dhidden_depth $dn_hidden_w $dn_hidden_d $dbaseLr $elr $dlayer
done