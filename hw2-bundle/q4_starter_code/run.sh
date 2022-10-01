python train.py --dataset=cora --dropout=0.5 --weight_decay=5e-3 --epochs=500
python train.py --dataset=enzymes --weight_decay=5e-3 --num_layers=3 --epochs=500

python -u train.py --dataset=cora --dropout=0.5 --weight_decay=5e-3 --epochs=500 --model_type=myGCN | tee Cora_GCN.log
python -u train.py --dataset=cora --dropout=0.5 --weight_decay=5e-3 --epochs=500 --model_type=GraphSage | tee Cora_GraphSage.log
python -u train.py --dataset=cora --dropout=0.5 --weight_decay=5e-3 --epochs=500 --model_type=GAT | tee Cora_GAT.log
python -u train.py --dataset=enzymes --weight_decay=5e-3 --num_layers=3 --epochs=500 --model_type=myGCN | tee ENZYMES_GCN.log
python -u train.py --dataset=enzymes --weight_decay=5e-3 --num_layers=3 --epochs=500 --model_type=GraphSage | tee ENZYMES_GraphSage.log
python -u train.py --dataset=enzymes --weight_decay=5e-3 --num_layers=3 --epochs=500 --model_type=GAT | tee ENZYMES_GAT.log