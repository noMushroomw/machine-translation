# machine-translation

LSTM Encoder-Decoder model for machine translation from French to English

```
python lstm.py
```

Bi-directional LSTM Encoder-Decoder model for machine translation from French to English

```
python bi-lstm.py
```

Multihead Attention Encoder-Decoder model for machine translation from French to English

```
python transformer.py
```

## default hyperparameters

--hidden_size 256 \
--n_iters 100000 \
 --print_every 5000 \
 --checkpoint_every 10000 \
 --initial_learning_rate 0.001 \
 --src_lang fr \
 --tgt_lang en \
 --train_file data/fren.train.bpe \
 --dev_file data/fren.dev.bpe \
 --test_file data/fren.test.bpe \
 --out_file translations.txt
