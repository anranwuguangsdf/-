#/usr/bin/zsh

rm -rf runs

echo
echo "#####################################################  Detection ##############################################################"
python detection.py
echo
echo "#####################################################  Segmentaion ##############################################################"
python segmentation.py
echo
echo "#####################################################  Classification ##############################################################"
python classification.py