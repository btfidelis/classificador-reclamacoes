import pickle

class Model:
    def __init__(self):
        self.models = ['count_vec', 'tdifd_trans', 'cp_a_']
    
    def _get_from_pickle(self, f_name):
        with open(f_name + '.pkl', 'rb') as m:
            return pickle.load(m)

    def _load(self):
        return Classifier(*[self._get_from_pickle(m) for m in self.models])

    def predict(self, sent):
        clf = self._load()
        print('Categoria: {}'.format(clf.predict([sent])))
        print(clf.model.classes_)
        print(clf.predict_prob([sent]))  


class Classifier:
    def __init__(self, count_vectorizer, tfidf_transformer, model):
        self.count_vectorizer = count_vectorizer
        self.tfidf_transformer = tfidf_transformer
        self.model = model

    def _prepare_sents(self, sents):
        return self.tfidf_transformer.transform(self.count_vectorizer.transform(sents))
    
    def predict(self, sents):
        tfidf_sents = self._prepare_sents(sents)
        return self.model.predict(tfidf_sents)

    def predict_prob(self, senteses):
        tfidf_sents = self._prepare_sents(senteses)
        return self.model.predict_proba(tfidf_sents)
    