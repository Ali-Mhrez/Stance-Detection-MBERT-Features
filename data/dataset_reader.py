import os
import json

stance_to_int = {'Agree': 0, 'Disagree': 1, 'Discuss': 2, 'Unrelated': 3, \
                 'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3, \
                 'other': 2} # some datasets merge discuss and unrelated

class AraStanceReader:
    """A class to read and process the AraStance dataset from a JSONL file.
    This dataset contains claims, articles, stances, and indices linking articles to claims.
    
    Args:
        path (str): Path to the dataset directory containing the JSONL file.
    
    Attributes:
        claims (list): List of claims.
        articles (list): List of articles.
        stances (list): List of stances corresponding to the claim/article pairs.
        article_claim (list): Indices linking each article to its corresponding claim.
    """
    def __init__(self, path):
        self.claims, self.articles, self.stances, self.article_claim = self.read(path)
        
    def read(self, path):
        """Reads the dataset from a JSONL file.
        
        Args:
            path (str): Path to the JSONL file containing the dataset.
        
        Returns:
            tuple: A tuple containing lists of claims, articles, stances, and article_claim indices.
        """
        claims, articles, stances = [], [], []
        article_claim = []
        with open(path, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                instance = json.loads(line)
                claims.append(instance['claim'])
                articles.extend(instance['article'])
                article_claim.extend([idx] * len(instance['article']))
                stances.extend(instance['stance'])
            assert len(stances) == len(articles) == len(article_claim)
            return claims, articles, stances, article_claim
        
    def __len__(self):
        """Returns the number of instances in the dataset."""
        return len(self.stances)
    
    def __getitem__(self, idx):
        """Returns the claim, article, and stance for a given index."""
        return (self.claims[self.article_claim[idx]], 
                self.articles[idx], 
                self.stances[idx])
        
        
class UnifiedFCReader:
    """A class to read and process the UnifiedFC dataset.
    This dataset contains claims, articles, stances, and indices linking articles to claims.
    
    Args:
        path (str): Path to the dataset directory containing the JSON files and train/test splits.
        folds (list, optional): List of fold names to read. Possible values are lists containing one or more of 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', and 'Fold_5'.
    
    Attributes:
        all_files (list): List of all files in the dataset directory.
        folds_dict (dict): Dictionary mapping fold names to lists of file names.
        claims (list): If folds are specified, this will contain claims from the selected folds.
        articles (list): If folds are specified, this will contain articles from the selected folds.
        stances (list): If folds are specified, this will contain stances corresponding to the claim/article pairs from the selected folds.
        article_claim (list): If folds are specified, this will contain indices linking each article to its corresponding claim from the selected folds.
    """
    def __init__(self, path, folds=None):
        self.all_files = os.listdir(os.path.join(path, 'dataset'))
        self.all_files = [file.replace(".json", "") for file in self.all_files]
        if ".DS_Store" in self.all_files:
            self.all_files.remove(".DS_Store")

        self.folds_dict = {}
        with open(os.path.join(path, 'train_test_splits.txt'), 'r') as file:
            lines = file.readlines()
            for line in lines:
                splitted = line.split(": ")
                self.folds_dict = {**self.folds_dict, **{splitted[0]: splitted[1].replace('\n', '').split(" ")}}
                
        if folds is not None:
            self.claims, self.articles, self.stances, self.article_claim = self.read_folds(path, folds)
                
    def read_folds(self, path, folds):
        """Reads the dataset from JSON files for specified folds.
        This method allows reading multiple folds of the dataset at once.
        
        Args:
            path (str): Path to the dataset directory containing the dataset.
            folds (list): List of fold names to read. Possible values are lists containing one or more of 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', and 'Fold_5'.
            
        Returns:
            tuple: A tuple containing lists of claims, articles, stances, and article_claim indices
        """
        assert isinstance(folds, list), "Folds should be a list of fold names."
        files = []
        for fold in folds:
            files.extend(self.folds_dict[fold])
        return self._read(path, files)

    def _read(self, path, files):
        """Reads the dataset from JSON files.

        Args:
            path (str): Path to the JSON files containing the dataset.
            files (list): List of files to read.

        Returns:
            tuple: A tuple containing lists of claims, articles, stances, and article_claim indices.
        """
        claims, articles, stances, article_claim = [], [], [], []
        for file in files:
            with open(os.path.join(path, "dataset", file+".json"), 'r', encoding='utf-8') as f:
                instance = json.loads(f.read())
                claims.append(instance['claim'])
                articles.extend(instance['article'])
                article_claim.extend([len(claims)-1] * len(instance['article']))
                stances.extend(instance['stance'])
        assert len(stances) == len(articles) == len(article_claim)
        return claims, articles, stances, article_claim
    
    def __len__(self):
        """Returns the number of instances in the dataset."""
        return len(self.stances)
    
    def __getitem__(self, idx):
        """Returns the claim, article, and stance for a given index."""
        return (self.claims[self.article_claim[idx]], 
                self.articles[idx], 
                self.stances[idx])