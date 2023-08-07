#sleep 30
rm -r "yolov5/runs/detect/"
rm -r "yolov5/runs/val/"
for (( c=1; c<=1; c++ ))
do
	python3 animals.py
	#sleep 30
done
cp -r "yolov5/runs/detect/" "animals/detect"
cp -r "yolov5/runs/val/" "animals/val"
