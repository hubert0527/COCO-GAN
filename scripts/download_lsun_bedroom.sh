git clone https://github.com/fyu/lsun.git
mkdir tmp
python ./lsun/download.py -o="./tmp/" --category="bedroom"
unzip ./tmp/bedroom_train_lmdb.zip -d ./tmp/

mkdir ./data/lsun
python ./lsun/data.py export ./tmp/bedroom_train_lmdb --out_dir="./data/lsun/" --flat
# rm tmp -R
# rm lsun -Rf