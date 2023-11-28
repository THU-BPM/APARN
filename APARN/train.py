import json
import os
import sys
import copy
import random
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW
from models.relational_attention_bert import RelationalAttentionBertClassifier
from data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix, Tokenizer4BertGCN, ABSAGCNData
from prepare_vocab import VocabHelp

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Instructor:
    ''' Model training and evaluation '''

    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4BertGCN(opt.max_length, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
            trainset = ABSAGCNData(opt.dataset_file['train'], tokenizer, opt=opt)
            testset = ABSAGCNData(opt.dataset_file['test'], tokenizer, opt=opt)
            if opt.dataset_file.get('valid'):
                validset = ABSAGCNData(opt.dataset_file['valid'], tokenizer, opt=opt)
            else:
                validset = testset
        else:
            tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_length=opt.max_length,
                data_file='{}/{}_tokenizer.dat'.format(opt.vocab_dir, opt.dataset))
            embedding_matrix = build_embedding_matrix(
                vocab=tokenizer.vocab,
                embed_dim=opt.embed_dim,
                data_file='{}/{}d_{}_embedding_matrix.dat'.format(opt.vocab_dir, str(opt.embed_dim), opt.dataset))

            logger.info("Loading vocab...")
            token_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_tok.vocab')  # token
            post_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_post.vocab')  # position
            pos_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pos.vocab')  # POS
            amr_edge_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_amr_edge.vocab')  # amr edge
            dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')  # deprel
            pol_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pol.vocab')  # polarity
            logger.info(
                "token_vocab: {}, post_vocab: {}, pos_vocab: {}, amr_edge_vocab: {}, dep_vocab: {}, pol_vocab: {}"
                .format(len(token_vocab), len(post_vocab), len(pos_vocab), len(amr_edge_vocab), len(dep_vocab),
                        len(pol_vocab)))

            # opt.tok_size = len(token_vocab)
            opt.post_size = len(post_vocab)
            opt.pos_size = len(pos_vocab)
            opt.amr_edge_size = len(amr_edge_vocab)

            vocab_help = (post_vocab, pos_vocab, dep_vocab, pol_vocab)
            self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
            trainset = SentenceDataset(opt.dataset_file['train'], tokenizer, opt=opt, vocab_help=vocab_help)
            testset = SentenceDataset(opt.dataset_file['test'], tokenizer, opt=opt, vocab_help=vocab_help)
            if opt.dataset_file.get('valid'):
                validset = SentenceDataset(opt.dataset_file['valid'], tokenizer, opt=opt, vocab_help=vocab_help)
            else:
                validset = testset

        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(dataset=validset, batch_size=opt.batch_size)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')

        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)  # xavier_uniform_
                else:
                    stdv = 1. / (p.shape[0] ** 0.5)
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def get_bert_optimizer(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        diff_part = ["bert.embeddings", "bert.encoder"]
        betas = (0.9, self.opt.adam_beta)

        if self.opt.diff_lr:
            logger.info("layered learning rate on")
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.bert_lr
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               not any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": self.opt.weight_decay,
                    "lr": self.opt.learning_rate
                },
                {
                    "params": [p for n, p in model.named_parameters() if
                               any(nd in n for nd in no_decay) and not any(nd in n for nd in diff_part)],
                    "weight_decay": 0.0,
                    "lr": self.opt.learning_rate
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, betas=betas, eps=self.opt.adam_epsilon)

        else:
            logger.info("bert learning rate on")
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.opt.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(
                    nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, betas=betas, eps=self.opt.adam_epsilon)

        return optimizer

    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs, _ = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                loss = criterion(outputs, targets)

                # if self.opt.losstype == "distance":
                #     loss = criterion(outputs, targets)
                # else:
                #     loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate(valid=True)
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('./APARN/state_dict'):
                                os.mkdir('./APARN/state_dict')
                            model_path = './APARN/state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}' \
                                .format(self.opt.model_name, self.opt.dataset, test_acc, f1)
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> saved: {}'.format(model_path))
                    if f1 > max_f1:
                        max_f1 = f1
                    logger.info('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'
                                .format(loss.item(), train_acc, test_acc, f1))
        return max_test_acc, max_f1, model_path

    def _evaluate(self, show_results=False, valid=False):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None
        detail_result = True
        detail_results = []
        if valid:
            dataset = self.valid_dataloader
        else:
            dataset = self.test_dataloader
        with torch.no_grad():
            for batch, sample_batched in enumerate(dataset):
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                outputs, penal = self.model(inputs)
                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        labels = targets_all.data.cpu()
        predic = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predic, digits=4)
            confusion = metrics.confusion_matrix(labels, predic)
            return report, confusion, test_acc, f1

        return test_acc, f1

    def _test(self, train_final_test=True):
        if train_final_test:
            self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True, valid=False)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion)

    def run(self):
        criterion = nn.CrossEntropyLoss()
        if 'bert' not in self.opt.model_name:
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
        else:
            optimizer = self.get_bert_optimizer(self.model)
        max_test_acc_overall = 0
        max_f1_overall = 0
        if 'bert' not in self.opt.model_name:
            self._reset_params()
        if not self.opt.model_path:
            max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)
            logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
            max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
            max_f1_overall = max(max_f1, max_f1_overall)
            torch.save(self.best_model.state_dict(), model_path)
            logger.info('>> saved: {}'.format(model_path))
            logger.info('#' * 60)
            logger.info('~!@max_test_acc_overall:{}'.format(max_test_acc_overall))
            logger.info('max_f1_overall:{}'.format(max_f1_overall))
            self._test()
        else:
            self.model.load_state_dict(torch.load(self.opt.model_path))
            self._test(train_final_test=False)


def main():
    model_classes = {
        'relationalbert': RelationalAttentionBertClassifier,
    }

    dataset_files = {
        'restaurant': {
            'train': './APARN/dataset/Restaurants_corenlp/train.json',
            'test': './APARN/dataset/Restaurants_corenlp/test.json',
        },
        'laptop': {
            'train': './APARN/dataset/Laptops_corenlp/train.json',
            'test': './APARN/dataset/Laptops_corenlp/test.json'
        },
        'twitter': {
            'train': './APARN/dataset/Tweets_corenlp/train.json',
            'test': './APARN/dataset/Tweets_corenlp/test.json',
        },
        'laptop2': {
            'train': './APARN/dataset/Laptops/train.json',
            'test': './APARN/dataset/Laptops/test.json'
        },
        'mams': {
            'train': './APARN/dataset/MAMS/train.json',
            'valid': './APARN/dataset/MAMS/valid.json',
            'test': './APARN/dataset/MAMS/test.json'
        },
        'restaurant_amr': {
            'train': './APARN/dataset/Restaurants_corenlp/train_with_amr.json',
            'test': './APARN/dataset/Restaurants_corenlp/test_with_amr.json',
        },
        'laptop_amr': {
            'train': './APARN/dataset/Laptops_corenlp/train_with_amr.json',
            'test': './APARN/dataset/Laptops_corenlp/test_with_amr.json'
        },
        'twitter_amr': {
            'train': './APARN/dataset/Tweets_corenlp/train_with_amr.json',
            'test': './APARN/dataset/Tweets_corenlp/test_with_amr.json',
        },
        'laptop2_amr': {
            'train': './APARN/dataset/Laptops/train_with_amr.json',
            'test': './APARN/dataset/Laptops/test_with_amr.json'
        },
        'mams_amr': {
            'train': './APARN/dataset/MAMS_corenlp/train_with_amr.json',
            'valid': './APARN/dataset/MAMS_corenlp/valid_with_amr.json',
            'test': './APARN/dataset/MAMS_corenlp/test_with_amr.json'
        },
        'r2l': {
            'train': './APARN/dataset/Restaurants_corenlp/train_with_amr.json',
            'test': './APARN/dataset/Laptops_corenlp/test_with_amr.json',
        },
        'r2t': {
            'train': './APARN/dataset/Restaurants_corenlp/train_with_amr.json',
            'test': './APARN/dataset/Tweets_corenlp/test_with_amr.json',
        },
        'l2r': {
            'train': './APARN/dataset/Laptops_corenlp/train_with_amr.json',
            'test': './APARN/dataset/Restaurants_corenlp/test_with_amr.json',
        },
        'l2t': {
            'train': './APARN/dataset/Laptops_corenlp/train_with_amr.json',
            'test': './APARN/dataset/Tweets_corenlp/test_with_amr.json',
        },
        't2r': {
            'train': './APARN/dataset/Tweets_corenlp/train_with_amr.json',
            'test': './APARN/dataset/Restaurants_corenlp/test_with_amr.json',
        },
        't2l': {
            'train': './APARN/dataset/Tweets_corenlp/train_with_amr.json',
            'test': './APARN/dataset/Laptops_corenlp/test_with_amr.json',
        },
    }

    input_colses = {
        'relationalbert': ['text_bert_indices', 'bert_segments_ids', 'attention_mask', 'asp_start', 'asp_end',
                           'adj_matrix', 'edge_adj', 'src_mask', 'aspect_mask'],
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }

    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='APARN', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='laptop', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--learning_rate', default=0.002, type=float)
    parser.add_argument('--l2reg', default=1e-4, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--post_dim', type=int, default=30, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=50, help='GCN mem dim.')
    parser.add_argument('--num_layers', type=int, default=1, help='Num of layers.')
    parser.add_argument('--polarities_dim', default=3, type=int, help='3')

    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
    parser.add_argument('--attn_dropout', type=float, default=0.3, help='Attention layer dropout rate.')
    parser.add_argument('--gcn_dropout', type=float, default=0.1, help='Attention layer dropout rate.')
    parser.add_argument('--lower', default=True, help='Lowercase all words.')
    parser.add_argument('--direct', default=False, help='directed graph or undirected graph')
    parser.add_argument('--loop', default=True)

    parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=50, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')

    parser.add_argument('--attention_heads', default=8, type=int, help='number of multi-attention heads')
    parser.add_argument('--dim_heads', default=64, type=int, help='dim of every head in multi-attention heads')
    parser.add_argument('--max_length', default=85, type=int)
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--vocab_dir', type=str, default='')
    parser.add_argument('--pad_id', default=0, type=int)
    parser.add_argument('--parseadj', default=False, action='store_true', help='dependency probability')
    parser.add_argument('--parsehead', default=False, action='store_true', help='dependency tree')
    parser.add_argument('--parseamr', default=False, action='store_true', help='abstract meaning tree')
    parser.add_argument('--cuda', default='0', type=str)
    parser.add_argument('--losstype', default=None, type=str, help="")
    parser.add_argument('--alpha', default=0.25, type=float)
    parser.add_argument('--beta', default=0.25, type=float)

    # * bert
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument("--adam_beta", default=0.999, type=float, help="Beta for Adam optimizer.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.4, help='BERT dropout rate.')
    parser.add_argument('--diff_lr', default=False, action='store_true')
    parser.add_argument('--bert_lr', default=3e-5, type=float)
    parser.add_argument('--part', default=1, type=float)
    parser.add_argument('--edge', default='normal', type=str, help="['normal', 'random', 'same']")
    parser.add_argument('--feature_type', default='1+A', type=str, help="['1+2+A', '1+2+A/3', '1+A']")
    parser.add_argument('--model_path', type=str, default='')
    opt = parser.parse_args()
    opt.amr_edge_stoi = './APARN/stoi.pt'
    opt.amr_edge_pt = './APARN/embedding.pt'
    opt.amr_edge_dim = 1024
    opt.edge_dropout = opt.bert_dropout
    opt.final_dropout = opt.bert_dropout

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]

    print("choice cuda:{}".format(opt.cuda))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    # opt.device = torch.device('cpu')

    # set random seed
    setup_seed(opt.seed)

    if not os.path.exists('./APARN/log'):
        os.makedirs('./APARN/log', mode=0o777)
    log_file = '{}-{}-{}.log0'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    logger.addHandler(logging.FileHandler("%s/%s" % ('./APARN/log', log_file)))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
