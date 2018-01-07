#!/bin/sh
python runRNNMusic.py ../data/music/BPROut/music_first2010-2014.txttrain_BPROutput.oneout.item64 music.item_embedding.RNN.64 music.user.embedding.RNN.64
python runRNNMusic.py ../data/music/BPROut/music_first2010-2014.txttrain_BPROutput.oneout.item128 music.item_embedding.RNN.128 music.user.embedding.RNN.128
python runRNNMusic.py ../data/music/BPROut/music_first2010-2014.txttrain_BPROutput.oneout.item256 music.item_embedding.RNN.256 music.user.embedding.RNN.256
python runMN2.py ../data/music/BPROut/music_first2010-2014.txttrain_BPROutput.oneout.item128 ../data/music/TransE/music4MN.TransEOutput music.item_embedding.in128.out256 music.user.embedding.in128.out256 256
python runMN2.py ../data/music/BPROut/music_first2010-2014.txttrain_BPROutput.oneout.item256 ../data/music/TransE/music4MN.TransEOutput music.item_embedding.in256.out256 music.user.embedding.in256.out256 256
python runMN2.py ../data/music/BPROut/music_first2010-2014.txttrain_BPROutput.oneout.item64 ../data/music/TransE/music4MN.TransEOutput music.item_embedding.in64.out256 music.user.embedding.in64.out256 256
python runMN2.py ../data/music/BPROut/music_first2010-2014.txttrain_BPROutput.oneout.item64 ../data/music/TransE/music4MN.TransEOutput music.item_embedding.in64.out128 music.user.embedding.in64.out128 128
python runMN2.py ../data/music/BPROut/music_first2010-2014.txttrain_BPROutput.oneout.item64 ../data/music/TransE/music4MN.TransEOutput music.item_embedding.in64.out64 music.user.embedding.in64.out64 64
python run_KVMN.py ../data/music/BPROut/music_first2010-2014.txttrain_BPROutput.oneout.item64 ../data/music/TransE/music4MN.TransEOutput ../data/music/TransE/music_selected_rel.txt music.item_embedding.KV.in64.out64 music.user.embedding.KV.in64.out64 64
python run_KVMN.py ../data/music/BPROut/music_first2010-2014.txttrain_BPROutput.oneout.item128 ../data/music/TransE/music4MN.TransEOutput ../data/music/TransE/music_selected_rel.txt music.item_embedding.KV.in128.out128 music.user.embedding.KV.in128.out128 128
python run_KVMN.py ../data/music/BPROut/music_first2010-2014.txttrain_BPROutput.oneout.item256 ../data/music/TransE/music4MN.TransEOutput ../data/music/TransE/music_selected_rel.txt music.item_embedding.KV.in256.out256 music.user.embedding.KV.in256.out256 256
