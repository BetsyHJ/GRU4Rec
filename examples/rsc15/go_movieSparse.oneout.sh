#!/bin/sh
python runRNNMovie.py ../data/movie_sparse/BPROut/ratings.dattrain_BPR.oneout.item64 item_embedding.RNN.64 user.embedding.RNN.64
python runRNNMovie.py ../data/movie_sparse/BPROut/ratings.dattrain_BPR.oneout.item128 item_embedding.RNN.128 user.embedding.RNN.128
python runRNNMovie.py ../data/movie_sparse/BPROut/ratings.dattrain_BPR.oneout.item256 item_embedding.RNN.256 user.embedding.RNN.256
python runMN2.py ../data/movie_sparse/BPROut/ratings.dattrain_BPR.oneout.item64 ../data/movie_sparse/TransE/movie4MN.TransEOutput item_embedding.in64.out64 user.embedding.in64.out64 64
python runMN2.py ../data/movie_sparse/BPROut/ratings.dattrain_BPR.oneout.item64 ../data/movie_sparse/TransE/movie4MN.TransEOutput item_embedding.in64.out128 user.embedding.in64.out128 128
python runMN2.py ../data/movie_sparse/BPROut/ratings.dattrain_BPR.oneout.item64 ../data/movie_sparse/TransE/movie4MN.TransEOutput item_embedding.in64.out256 user.embedding.in64.out256 256
python runMN2.py ../data/movie_sparse/BPROut/ratings.dattrain_BPR.oneout.item128 ../data/movie_sparse/TransE/movie4MN.TransEOutput item_embedding.in128.out256 user.embedding.in128.out256 256
python runMN2.py ../data/movie_sparse/BPROut/ratings.dattrain_BPR.oneout.item256 ../data/movie_sparse/TransE/movie4MN.TransEOutput item_embedding.in256.out256 user.embedding.in256.out256 256
python run_KVMN.py ../data/movie_sparse/BPROut/ratings.dattrain_BPR.oneout.item64 ../data/movie_sparse/TransE/movie4MN.TransEOutput ../data/movie_sparse/TransE/movie_selected_rel.txt item_embedding.KV.in64.out64 user.embedding.KV.in64.out64 64
python run_KVMN.py ../data/movie_sparse/BPROut/ratings.dattrain_BPR.oneout.item256 ../data/movie_sparse/TransE/movie4MN.TransEOutput ../data/movie_sparse/TransE/movie_selected_rel.txt item_embedding.KV.in256.out256 user.embedding.KV.in256.out256 256
