#!/bin/sh
#python runRNNBook.py ../data/book/BPROut/book.ratingtrain_BPROutput.oneout.item64 book.item_embedding.RNN.64 book.user.embedding.RNN.64
#python runRNNBook.py ../data/book/BPROut/book.ratingtrain_BPROutput.oneout.item128 book.item_embedding.RNN.128 book.user.embedding.RNN.128
#python runRNNBook.py ../data/book/BPROut/book.ratingtrain_BPROutput.oneout.item256 book.item_embedding.RNN.256 book.user.embedding.RNN.256
python runMN2.py ../data/book/BPROut/book.ratingtrain_BPROutput.oneout.item64 ../data/book/TransE/book4MN.TransEOutput book.item_embedding.in64.out64 book.user.embedding.in64.out64 64
python runMN2.py ../data/book/BPROut/book.ratingtrain_BPROutput.oneout.item64 ../data/book/TransE/book4MN.TransEOutput book.item_embedding.in64.out128 book.user.embedding.in64.out128 128
python runMN2.py ../data/book/BPROut/book.ratingtrain_BPROutput.oneout.item64 ../data/book/TransE/book4MN.TransEOutput book.item_embedding.in64.out256 book.user.embedding.in64.out256 256
python runMN2.py ../data/book/BPROut/book.ratingtrain_BPROutput.oneout.item128 ../data/book/TransE/book4MN.TransEOutput book.item_embedding.in128.out256 book.user.embedding.in128.out256 256
python runMN2.py ../data/book/BPROut/book.ratingtrain_BPROutput.oneout.item256 ../data/book/TransE/book4MN.TransEOutput book.item_embedding.in256.out256 book.user.embedding.in256.out256 256
