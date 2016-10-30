cd model1
echo 'run model 1 - data driven model'
python main_model1.py
echo 'model 1 export complete'
cd ../model2/script
echo 'run model 2 - understanding driven model'
python main_model2.py
echo 'model 2 export complete'
cd ../..
echo 'model ensemble begin'
python ensemble.py
echo 'model ensemble complete'