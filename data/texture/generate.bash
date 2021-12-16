#! /bin/bash

for i in 1 2 3 4 5 6 7 8 9 10
do
	convert -size $((2**i))x$((2**i)) xc:rgba\(121,122,123,0.1\) $((2**i))x$((2**i)).png
done
