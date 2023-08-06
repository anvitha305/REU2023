sleep 30
for (( c=1; c<=10; c++ ))
do
	python3 mnist.py
	mv mnist_local.png "mnist/mnist_local${c}.png"
	mv mnist_stats.txt "mnist/mnist_stats${c}.txt"
	sleep 30
done
