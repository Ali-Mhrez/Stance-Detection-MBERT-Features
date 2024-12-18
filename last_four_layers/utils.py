import json

stance_to_int = {'Agree': 0, 'Disagree': 1, 'Discuss': 2, 'Unrelated': 3, \
                 'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3, \
                 'other': 2} # some datasets merge discuss and unrelated

class AraStanceData:
  """
  Class for AraStance dataset
  """

  def __init__(self, path):
    """
    Args:
      path: str, path/filename.extension
    """
    self.claims, self.articles, self.stances, self.article_claim = self.read(path)
    
  def read(self, path):
    """
    Read Arastance data from jsonl files
    Args:
      path: str, path/filename.extension
    Returns:
      claims: list, all claims
      articles: list, all articles
      stances: list, articles stances
      article_claim: list, mapping between articles and claims
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
    return claims, articles, stances, article_claim