import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")  # Add this line

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PreProcessor():

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
    

    def load_requirements(self):
        nfrs = []
        frs = []
        PATH = 'Functional_Requirements_Fun.txt'

        with open(PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("NFR"):
                    nfrs.append(line)
                elif line.startswith("FR"):
                    frs.append(line)
        
        print(f"Loaded {len(nfrs)} NFRs and {len(frs)} FRs")  # Add this debug line
        return nfrs, frs
    
    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            return wordnet.NOUN



    def preprocess_requirement(
        self,
        requirement,
        tokenize=True,
        remove_stopwords=True,
        use_pos=True,
        lemmatize=True,
        stem=False,
    ):
        # Remove requirement ID
        requirement = re.sub(r"^[A-Z]+[0-9]+[: ]+", "", requirement)
        requirement = requirement.lower()

        # Tokenization
        tokens = word_tokenize(requirement) if tokenize else requirement.split()

        # Remove non-alphabetic tokens
        tokens = [t for t in tokens if t.isalpha()]

        # Stop-word removal
        if remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        # POS tagging
        if use_pos:
            pos_tags = nltk.pos_tag(tokens)
        else:
            pos_tags = [(t, None) for t in tokens]

        # Lemmatization or stemming
        if lemmatize:
            tokens = [
                self.lemmatizer.lemmatize(w, self.get_wordnet_pos(p)) if p else self.lemmatizer.lemmatize(w)
                for w, p in pos_tags
            ]
        elif stem:
            tokens = [self.stemmer.stem(w) for w, _ in pos_tags]
        else:
            tokens = [w for w, _ in pos_tags]

        return " ".join(tokens)


    def pre_process(self, pre_prcessing_config: dict):
        nfrs, frs = self.load_requirements()

        nfr_texts = [self.preprocess_requirement(nfr, **pre_prcessing_config) for nfr in nfrs]
        fr_texts  = [self.preprocess_requirement(fr, **pre_prcessing_config) for fr in frs]

        return nfr_texts, fr_texts
    
class RequirementsTracer():

    def __init__(self):
        self.pre_processor = PreProcessor()
        self.pre_processing_configs = [
            dict(tokenize=True, remove_stopwords=True, use_pos=True, lemmatize=True, stem=False), # All + lemmatize
            dict(tokenize=True, remove_stopwords=True, use_pos=True, lemmatize=False, stem=True), # All + stem
            dict(tokenize=True, remove_stopwords=False, use_pos=True, lemmatize=False, stem=True) # No stop word removal + stem
        ]
        self.THRESHOLD = 0.2

    def vectorizer(self, nfr_texts, fr_texts):
        vectorizer = TfidfVectorizer()

        all_texts = nfr_texts + fr_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        nfr_vectors = tfidf_matrix[:len(nfr_texts)]
        fr_vectors  = tfidf_matrix[len(nfr_texts):]

        return nfr_vectors, fr_vectors
    
    def print_trace(self, trace_matrix, variation: int):
        nfrs, frs = self.pre_processor.load_requirements()
        output_path = f"trace_matrix_{str(variation + 1)}.txt"

        with open(output_path, "w", encoding="utf-8") as f:
            for fr, row in zip(frs, trace_matrix):
                fr_id = fr.split(":")[0]
                # Write FR id followed by all columns in the trace row
                row_vals = ",".join(map(str, row.tolist()))
                f.write(f"{fr_id},{row_vals}\n")

    def generate_trace_matrix(self, pre_processing_config, variation):
        nfr_texts, fr_texts = self.pre_processor.pre_process(pre_processing_config)

        nfr_vectors, fr_vectors = self.vectorizer(nfr_texts, fr_texts)

        similarity_matrix = cosine_similarity(fr_vectors, nfr_vectors)

        trace_matrix = (similarity_matrix >= self.THRESHOLD).astype(int)

        self.print_trace(trace_matrix, variation)


    def trace_requirements(self):
        for i in range(3):
            self.generate_trace_matrix(self.pre_processing_configs[i], variation=i)


if __name__ == '__main__':
    requirements_tracer = RequirementsTracer()
    requirements_tracer.trace_requirements()
