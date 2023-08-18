import argparse
from utils import input_from_file
from utils import output_to_file

def remove_baseline(data, baseline_p) :
    # Remove baseline
    X = data[..., baseline_p:]
    return X


if __name__ == '__main__':
    # processing arguments #######################################################
    parser = argparse.ArgumentParser(description='preprocessor for ~~~')

    ## file path
    parser.add_argument('-i', '--input-dir', type=str,
                        help='directory path that contains input files')
    parser.add_argument('-d', '--data-dir', type=str,
                        help='directory path in which output files will be saved')

    ## values
    parser.add_argument('-b', '--base-line', type=int,
                        help='subject index')

    parser.add_argument('--mode', type=str, default='train', 
                        help='preprocessing mode: choose one of train or test')

    ## parsing arguments
    args = parser.parse_args()

    # execute processing #########################################################

    use_call_train = False if args.mode == 'test' else True
    ret = False
    if not use_call_train:
        ret = input_from_file(args.input_dir, ['test_X'])
    else:
        ret = input_from_file(args.input_dir, ['train_X'])

    X=ret[0]

    base_line=args.base_line
    if base_line ==None : base_line= 100

    X = remove_baseline(X, base_line)

    ret = False
    if not use_call_train:
        ret = output_to_file(args.data_dir, ['test_X'], [X])
    else:
        ret = output_to_file(args.data_dir, ['train_X'], [X])

    if ret:
        exit(0)
    else:
        exit(-1)
