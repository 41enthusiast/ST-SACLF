import pandas
import os
import shutil
import glob

def load_labels(path):
    with open(path, newline='') as f:
        df = pandas.read_csv(f, sep=' ', header=None)
        fnames = list(df[0])
        tgts = list(df[1])
    return fnames, tgts

if __name__ == '__main__':
    data_root = '../data/pacs_data'
    tgt_root = 'pacs_imagenet_style'
    splits = ['train', 'crossval', 'test']
    domain_names = ['art_painting', 'cartoon', 'photo', 'sketch']
    label_names = 'dog  elephant  giraffe  guitar  horse  house  person'.split('  ')
    for domain_name in domain_names:
        for data_type in splits:
            for lbl in label_names:
                os.makedirs(f'{tgt_root}/{domain_name}/{data_type}/{lbl}', exist_ok=True)
        print('Domain', domain_name)
        for data_type in splits:
            print('split', data_type)
            domain_label_path = glob.glob(f'../data/{domain_name}_{data_type}*')[0]
            domain_fnames, domain_tgts = load_labels(domain_label_path)
            for fpath, tgt in zip(domain_fnames, domain_tgts):
                fname= fpath.split('/')[-1]
                shutil.copyfile(f'{data_root}/{fpath}', f'{tgt_root}/{domain_name}/{data_type}/{label_names[tgt-1]}/{fname}')
