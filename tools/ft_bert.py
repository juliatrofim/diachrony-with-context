import sys
import os

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("usage: {} <data_dir> <init_bert_ckpt> <bert_config_json>".format(sys.argv[0]))
        sys.exit(1)

    import tensorflow as tf

    data_dir = sys.argv[1]
    if data_dir[-1] == '/':
        data_dir = data_dir[:-1]

    os.system('git clone https://github.com/kwonmha/bert-vocab-builder')
    os.system('python3 bert-vocab-builder/subword_builder.py --corpus_filepattern "{}/*" --output_filename "vocab.txt"'.format(data_dir))

    input_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    os.system('git clone https://github.com/google-research/bert')
    os.system('python3 bert/create_pretraining_data.py \
              --input_file={}  \
              --output_file=bert_train_input.tfrecord \
              --vocab_file=vocab.txt \
              --do_lower_case=False \
              --max_seq_length=128 \
              --max_predictions_per_seq=20 \
              --masked_lm_prob=0.15 \
              --random_seed=42 \
              --dupe_factor=5'.format(','.join(input_files)))

    init_bert = sys.argv[2]
    bert_config = sys.argv[3]

    print('save bert checkpoints to bert_output/')
    os.system('python3 bert/run_pretraining.py \
              --input_file=bert_train_input.tfrecord \
              --output_dir=bert_output \
              --do_train=True \
              --do_eval=True \
              --init_checkpoint={} \
              --bert_config_file={} \
              --train_batch_size=32 \
              --max_seq_length=128 \
              --max_predictions_per_seq=20 \
              --num_train_steps=1000 \
              --num_warmup_steps=10 \
              --learning_rate=2e-5'.format(init_bert, bert_config))
