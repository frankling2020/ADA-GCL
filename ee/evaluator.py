import torch
import numpy as np
from torch import nn
from torch.optim import Adam

from sklearn.metrics import f1_score
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.svm import LinearSVC, SVC

from abc import ABC, abstractmethod
from tqdm import tqdm


def get_split(num_samples: int, train_ratio: float = 0.1, test_ratio: float = 0.8):
    assert train_ratio + test_ratio < 1
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'valid': indices[train_size: test_size + train_size],
        'test': indices[test_size + train_size:]
    }


def split_to_numpy(x, y, split):
    keys = ['train', 'test', 'valid']
    objs = [x, y]
    return [obj[split[key]].detach().cpu().numpy() for obj in objs for key in keys]


def get_predefined_split(x_train, x_val, y_train, y_val, return_array=True):
    test_fold = np.concatenate([-np.ones_like(y_train), np.zeros_like(y_val)])
    ps = PredefinedSplit(test_fold)
    if return_array:
        x = np.concatenate([x_train, x_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        return ps, [x, y]
    return ps


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        pass

    def __call__(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        for key in ['train', 'test', 'valid']:
            assert key in split

        result = self.evaluate(x, y, split)
        return result


class BaseSKLearnEvaluator(BaseEvaluator):
    def __init__(self, evaluator, params):
        self.evaluator = evaluator
        self.params = params

    def evaluate(self, x, y, split):
        x_train, x_test, x_val, y_train, y_test, y_val = split_to_numpy(
            x, y, split)
        ps, [x_train, y_train] = get_predefined_split(x_train, x_val, y_train, y_val)
        classifier = GridSearchCV(
            self.evaluator, self.params, cv=ps, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        
        train_micro = f1_score(
            y_train, classifier.predict(x_train), average='micro')
        val_micro = f1_score(
            y_val, classifier.predict(x_val), average='micro')
        test_micro = f1_score(
            y_test, classifier.predict(x_test), average='micro')

        return {
            'test': test_micro,
            'train': train_micro,
            'valid': val_micro,
        }


class SVMEvaluator(BaseSKLearnEvaluator):
    def __init__(self, linear=True, params=None):
        if linear:
            self.evaluator = LinearSVC(dual=False, max_iter=10000)
        else:
            self.evaluator = SVC(max_iter=100000)
        if params is None:
            params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        super(SVMEvaluator, self).__init__(self.evaluator, params)


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 20):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(),
                         lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_train_micro = 0
        best_val_micro = 0
        best_test_micro = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_pred = classifier(
                        x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_micro = f1_score(y_test, y_pred, average='micro')

                    y_val = y[split['valid']].detach().cpu().numpy()
                    y_pred = classifier(
                        x[split['valid']]).argmax(-1).detach().cpu().numpy()
                    val_micro = f1_score(y_val, y_pred, average='micro')


                    y_train = y[split['train']].detach().cpu().numpy()
                    y_pred = classifier(
                        x[split['train']]).argmax(-1).detach().cpu().numpy()
                    train_micro = f1_score(y_train, y_pred, average='micro')

                    if val_micro >= best_val_micro:
                        best_train_micro = train_micro
                        best_val_micro = val_micro
                        best_test_micro = test_micro
                        # best_epoch = epoch

                    pbar.set_postfix(
                        {'best test F1Mi': best_test_micro, 'valid F1Mi': best_val_micro})
                    pbar.update(self.test_interval)

        return {
            'test': best_test_micro,
            'train': best_train_micro,
            'valid': best_val_micro
        }
