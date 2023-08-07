sleep 30
rm -r "yolov5/runs/detect/"
rm -r "yolov5/runs/val/"
for (( c=1; c<=10; c++ ))
do
	python3 solar.py
	sleep 30
done
mkdir "solar"
mkdir "solar/detect"
mkdir "solar/val"
cp -r "yolov5/runs/detect/" "solar/detect"
cp -r "yolov5/runs/val/" "solar/val"
