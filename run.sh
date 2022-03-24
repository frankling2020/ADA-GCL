device='cpu'
batch_size=256
tau=0.2
log='log.txt'
epochs=500
hidden_dims=32
num_layers=2

for dataset in "MUTAG" "PROTEINS" "NCI1" "IMDB-BINARY" "IMDB-MULTI" "DD"
do
    python gcl.py --dataset=$dataset --batch_size=$batch_size --tau=$tau --log=$log --device=$device --epochs=$epochs --hidden_dims=$hidden_dims --num_layers=$num_layers
done