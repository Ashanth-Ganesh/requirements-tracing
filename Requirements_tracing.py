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
nltk.download("averaged_perceptron_tagger_eng")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
        
        print(f"Loaded {len(nfrs)} NFRs and {len(frs)} FRs")
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

        return nfr_vectors, fr_vectors, vectorizer
    
    def print_trace(self, trace_matrix, variation: int):
        nfrs, frs = self.pre_processor.load_requirements()
        output_path = f"trace_matrix_{str(variation + 1)}.txt"

        with open(output_path, "w", encoding="utf-8") as f:
            for fr, row in zip(frs, trace_matrix):
                fr_id = fr.split(":")[0]
                row_vals = ",".join(map(str, row.tolist()))
                f.write(f"{fr_id},{row_vals}\n")

    def generate_trace_matrix(self, pre_processing_config, variation):
        print("\n" + "="*80)
        print(f"VARIANT {variation + 1}")
        print("="*80)
        
        # Step 1: Get config details
        config_desc = []
        if pre_processing_config['tokenize']:
            config_desc.append("Tokenization")
        if pre_processing_config['remove_stopwords']:
            config_desc.append("Stop-word removal")
        if pre_processing_config['use_pos']:
            config_desc.append("POS tagging")
        if pre_processing_config['lemmatize']:
            config_desc.append("Lemmatization")
        if pre_processing_config['stem']:
            config_desc.append("Stemming")
        
        print(f"\nConfiguration: {', '.join(config_desc)}")
        print("-"*80)
        
        # Step 2: Pre-processing
        print("\n### STEP 1: PRE-PROCESSING ###")
        nfrs, frs = self.pre_processor.load_requirements()
        nfr_texts, fr_texts = self.pre_processor.pre_process(pre_processing_config)
        
        print(f"\nShowing first 3 examples:")
        for i in range(min(3, len(nfrs))):
            print(f"\nOriginal NFR: {nfrs[i]}")
            print(f"Processed NFR: {nfr_texts[i]}")
        
        for i in range(min(3, len(frs))):
            print(f"\nOriginal FR: {frs[i]}")
            print(f"Processed FR: {fr_texts[i]}")
        
        # Step 3: TF-IDF Vectorization
        print("\n### STEP 2: TF-IDF VECTORIZATION ###")
        nfr_vectors, fr_vectors, vectorizer = self.vectorizer(nfr_texts, fr_texts)
        
        print(f"\nVocabulary size: {len(vectorizer.vocabulary_)}")
        print(f"NFR vectors shape: {nfr_vectors.shape}")
        print(f"FR vectors shape: {fr_vectors.shape}")
        
        # Show top TF-IDF terms for first FR
        feature_names = vectorizer.get_feature_names_out()
        first_fr_vector = fr_vectors[0].toarray()[0]
        top_indices = first_fr_vector.argsort()[-10:][::-1]
        print(f"\nTop 10 TF-IDF terms for {frs[0].split(':')[0]}:")
        for idx in top_indices:
            if first_fr_vector[idx] > 0:
                print(f"  {feature_names[idx]}: {first_fr_vector[idx]:.4f}")
        
        # Step 4: Cosine Similarity
        print("\n### STEP 3: COSINE SIMILARITY CALCULATION ###")
        similarity_matrix = cosine_similarity(fr_vectors, nfr_vectors)
        
        print(f"\nSimilarity matrix shape: {similarity_matrix.shape}")
        print(f"(rows=FRs: {len(frs)}, columns=NFRs: {len(nfrs)})")
        
        # Step 5: Show top similarities for each FR
        print("\n### STEP 4: TOP SIMILARITIES (SORTED) ###")
        print(f"\nTop 5 NFR matches for first 3 FRs:")
        
        for i in range(min(3, len(frs))):
            fr_id = frs[i].split(":")[0]
            similarities = similarity_matrix[i]
            
            # Get top 5 indices
            top_5_indices = similarities.argsort()[-5:][::-1]
            
            print(f"\n{fr_id}:")
            for idx in top_5_indices:
                nfr_id = nfrs[idx].split(":")[0]
                score = similarities[idx]
                print(f"  {nfr_id}: {score:.4f} {'[TRACED]' if score >= self.THRESHOLD else ''}")
        
        # Step 6: Apply threshold
        print(f"\n### STEP 5: APPLYING THRESHOLD (>= {self.THRESHOLD}) ###")
        trace_matrix = (similarity_matrix >= self.THRESHOLD).astype(int)
        
        total_traces = np.sum(trace_matrix)
        print(f"\nTotal trace links found: {total_traces}")
        print(f"Average traces per FR: {total_traces / len(frs):.2f}")
        
        # Show trace statistics
        traces_per_fr = np.sum(trace_matrix, axis=1)
        print(f"FRs with 0 traces: {np.sum(traces_per_fr == 0)}")
        print(f"FRs with 1+ traces: {np.sum(traces_per_fr > 0)}")
        print(f"Max traces for single FR: {np.max(traces_per_fr)}")
        
        # Step 7: Save results
        self.print_trace(trace_matrix, variation)
        print(f"\nTrace matrix saved to: trace_matrix_{variation + 1}.txt")
        
        print("\n" + "="*80 + "\n")

    def trace_requirements(self):
        for i in range(3):
            self.generate_trace_matrix(self.pre_processing_configs[i], variation=i)
        
        print("\n" + "#"*80)
        print("ALL VARIANTS COMPLETED")
        print("#"*80)

if __name__ == '__main__':
    requirements_tracer = RequirementsTracer()
    requirements_tracer.trace_requirements()