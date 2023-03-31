#!/bin/bash
for shape in 8 16 32 64
do
    for stride in 1 2 3
    do
        for qubits in 4 9 12 16
        do
            for ((filters=1; filters<=$qubits; filters++))
            do
                kss=$(echo "sqrt($qubits)" | bc)
                
                for ((ks=2; ks <= $kss; ks++))
                do
                    echo "($shape,$shape,1)" $stride $qubits $filters $ks
                    python test_quanvolution.py --img_shape $shape $shape 1  --qubits $qubits --filters $filters --kernel_size $ks --stride $stride --num_jobs 16 --circuit 'ry' &>/dev/null
                done
            done
        done
    done
done