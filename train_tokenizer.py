import os
from tqdm import tqdm
from dataset import load_dataset
import sentencepiece as spm
import argparse

def trainTokenizer(vocab_size, dataset, save_path_prefix,tmp_file_save_path):
    #download dataset
    if dataset == "TinyStories":
        dataset = load_dataset("roneneldan/TinyStories")
    else:
        raise Exception("Given dataset is not available at the moment")
    
    total_text_file = os.path.join(tmp_file_save_path,"total_text.txt")
    print(f"merging the data to single temp text file at {total_text_file}")
    with open(total_text_file, "w", encoding="utf-8") as f:
        for row in tqdm(dataset["train"]):
            f.write(row["text"].strip() + '\n')
    prefix = os.path.join(save_path_prefix,f"tok_{vocab_size}")
    print(f"Training the tokenizer and savign the model at {prefix}")
    spm.SentencePieceTrainer.train(input=total_text_file,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size,
                                   self_test_sample_size=0,
                                   input_format="text",
                                   character_coverage=1.0,
                                   num_threads=os.cpu_count(),
                                   split_digits=True,
                                   allow_whitespace_only_pieces=True,
                                   byte_fallback=True,
                                   unk_surface=r" \342\201\207 ",
                                   normalization_rule_name="identity")
    print("Done!")
    os.remove(total_text_file)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", dest="vocab_size", required=True, help="Vocaulary size")
    parser.add_argument("--dataset", dest="dataset", required=False, default="TinyStories", 
                        help="Dataset on which model to train on")
    parser.add_argument("--save_path_prefix", dest="save_path_prefix", help="where to save the tokenizer")
    parser.add_argument("--tmp_file_save_path", dest = "tmp_file_save_path", required=False, default="saved_artifacts/datasets")
    args = parser.parse_args()
    # read the arguments
    vocab_size = args.vocab_size
    dataset = args.dataset
    save_path_prefix = args.save_path_prefix
    tmp_file_save_path = args.tmp_file_save_path
    # call the training funciton
    trainTokenizer(vocab_size, dataset, save_path_prefix,tmp_file_save_path)


