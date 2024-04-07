# Wordnet way of finding similarity between two sentences

# import nltk
# from nltk.corpus import wordnet as wn

# class SentenceSimilarity:
    
#     def __init__(self):
#         self.word_order = False
        
#     def identifyWordsForComparison(self, sentence):
#         # Tokenize and tag words, keep only nouns and verbs
#         tokens = nltk.word_tokenize(sentence)
#         pos_tags = nltk.pos_tag(tokens)
#         return [word for word, pos in pos_tags if pos.startswith('N') or pos.startswith('V')]
    
#     def wordSenseDisambiguation(self, words):
#         # Disambiguate word senses using WordNet
#         senses = []
#         for word in words:
#             synsets = wn.synsets(word)
#             if synsets:
#                 senses.append(synsets[0])  # Choose the first synset for simplicity
#         return senses
    
#     def calculateSimilarity(self, synsets1, synsets2):
#         # Calculate similarity using path similarity metric
#         similarity_scores = []
#         for synset1 in synsets1:
#             for synset2 in synsets2:
#                 similarity = synset1.path_similarity(synset2)
#                 if similarity is not None:
#                     similarity_scores.append(similarity)
#         if similarity_scores:
#             return sum(similarity_scores) / len(similarity_scores)
#         else:
#             return 0.0
    
#     def main(self, sentence1, sentence2):
#         words1 = self.identifyWordsForComparison(sentence1)
#         words2 = self.identifyWordsForComparison(sentence2)
#         synsets1 = self.wordSenseDisambiguation(words1)
#         synsets2 = self.wordSenseDisambiguation(words2)
#         similarity_score = self.calculateSimilarity(synsets1, synsets2)
#         print("Similarity score:", similarity_score)

# obj = SentenceSimilarity()
# # sentence1 = "jake ate apple."
# # sentence2 = "apple ate jake."
# sentence1 = "jake ate beef."
# sentence2 = "nobody ate."

# obj.main(sentence1, sentence2)


# wordnet version 2

# import nltk
# from nltk.corpus import wordnet as wn

# class SentenceSimilarity:
    
#     def __init__(self):
#         self.word_order = False
        
#     def identifyWordsForComparison(self, sentence):
#         # Tokenize and tag words, keep only nouns and verbs
#         tokens = nltk.word_tokenize(sentence)
#         pos_tags = nltk.pos_tag(tokens)
#         return [word for word, pos in pos_tags if pos.startswith('N') or pos.startswith('V')]
    
#     def wordSenseDisambiguation(self, words):
#         # Disambiguate word senses using WordNet
#         senses = []
#         for word in words:
#             synsets = wn.synsets(word)
#             if synsets:
#                 senses.append(synsets[0])  # Choose the first synset for simplicity
#         return senses
    
#     def calculateSimilarity(self, synsets1, synsets2):
#         # Calculate similarity using path similarity metric
#         similarity_scores = []
#         for synset1 in synsets1:
#             for synset2 in synsets2:
#                 similarity = synset1.path_similarity(synset2)
#                 if similarity is not None:
#                     similarity_scores.append(similarity)
#         if similarity_scores:
#             return sum(similarity_scores) / len(similarity_scores)
#         else:
#             return 0.0
    
#     def main(self, sentence1, sentence2):
#         words1 = self.identifyWordsForComparison(sentence1)
#         words2 = self.identifyWordsForComparison(sentence2)
#         synsets1 = self.wordSenseDisambiguation(words1)
#         synsets2 = self.wordSenseDisambiguation(words2)
#         similarity_score = self.calculateSimilarity(synsets1, synsets2)
#         print("Similarity score:", similarity_score)

# obj = SentenceSimilarity()
# # Example sentences
# sentence1 = "Jake ate an apple."
# sentence2 = "jake ate an apple"

# obj.main(sentence1, sentence2)


# transformer way of finding similarity between two sentences

# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity


# class SentenceSimilarity:
    
#     def __init__(self):
#         self.stop_words = set(stopwords.words('english'))
#         self.lemmatizer = WordNetLemmatizer()
    
#     def preprocess_sentence(self, sentence):
#         # Tokenize the sentence
#         words = word_tokenize(sentence)
#         # Remove stopwords and lemmatize the remaining words
#         words = [self.lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in self.stop_words]
#         # Join the preprocessed words back into a sentence
#         return ' '.join(words)
    
#     def calculate_similarity(self, sentence1, sentence2):
#         # Preprocess the sentences
#         processed_sentence1 = self.preprocess_sentence(sentence1)
#         processed_sentence2 = self.preprocess_sentence(sentence2)
        
#         # Compute TF-IDF vectors
#         vectorizer = TfidfVectorizer()
#         tfidf_matrix = vectorizer.fit_transform([processed_sentence1, processed_sentence2])
        
#         # Compute cosine similarity between the TF-IDF vectors
#         similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
#         return similarity_score

# # Example usage
# sentence1 = "jake ate beef."
# sentence2 = "nobody ate."

# similarity_obj = SentenceSimilarity()
# similarity_score = similarity_obj.calculate_similarity(sentence1, sentence2)
# print("Similarity score:", similarity_score)


# version 3


# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer

# # nltk.download('punkt')
# # nltk.download('stopwords')
# # nltk.download('wordnet')

# class SentenceSimilarity:
    
#     def __init__(self):
#         self.stop_words = set(stopwords.words('english'))
#         self.lemmatizer = WordNetLemmatizer()
    
#     def preprocess_sentence(self, sentence):
#         # Convert to lowercase
#         sentence = sentence.lower()
#         # Tokenize the sentence
#         words = word_tokenize(sentence)
#         # Remove punctuation and stopwords, and lemmatize the remaining words
#         words = [self.lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in self.stop_words]
#         return words
    
#     def calculate_similarity(self, sentence1, sentence2):
#         # Preprocess the sentences
#         words1 = set(self.preprocess_sentence(sentence1))
#         words2 = set(self.preprocess_sentence(sentence2))
        
#         # Calculate Jaccard similarity
#         intersection = len(words1.intersection(words2))
#         union = len(words1.union(words2))
#         similarity_score = intersection / union
#         return similarity_score

# # Example usage
# sentence1 = "Jake ate an apple."
# sentence2 = "apple ate a jake"


# similarity_obj = SentenceSimilarity()
# similarity_score = similarity_obj.calculate_similarity(sentence1, sentence2)
# print("Similarity score:", similarity_score)


# version 4

# import nltk
# from gensim.models import Word2Vec

# nltk.download('punkt')

# class SentenceSimilarity:
    
#     def __init__(self, model_path):
#         # Load pre-trained Word2Vec model
#         self.model = Word2Vec.load(model_path)
    
#     def get_sentence_embedding(self, sentence):
#         # Tokenize the sentence into words
#         tokens = nltk.word_tokenize(sentence)
#         # Filter out tokens not in the vocabulary
#         words = [word for word in tokens if word in self.model.wv]
#         # Return the mean of word embeddings as the sentence embedding
#         if words:
#             return sum(self.model.wv[word] for word in words) / len(words)
#         else:
#             return None
    
#     def calculate_similarity(self, sentence1, sentence2):
#         # Get embeddings for both sentences
#         embedding1 = self.get_sentence_embedding(sentence1)
#         embedding2 = self.get_sentence_embedding(sentence2)
#         # Check if both sentences contain embeddings
#         if embedding1 is not None and embedding2 is not None:
#             # Calculate cosine similarity between the embeddings
#             similarity_score = self.calculate_cosine_similarity(embedding1, embedding2)
#             return similarity_score
#         else:
#             # If one or both sentences don't contain embeddings, return None
#             return None
    
#     def calculate_cosine_similarity(self, vector1, vector2):
#         # Calculate cosine similarity between two vectors
#         similarity_score = sum(a * b for a, b in zip(vector1, vector2))
#         return similarity_score

# # Example usage
# sentence1 = "Jake ate an apple."
# sentence2 = "apple ate a jake."

# model_path = "path_to_pretrained_word2vec_model"  # Provide the path to the pre-trained Word2Vec model
# similarity_obj = SentenceSimilarity(model_path)
# similarity_score = similarity_obj.calculate_similarity(sentence1, sentence2)
# print("Similarity score:", similarity_score)


# compare summaries

## better version

# import torch
# from sentence_transformers import SentenceTransformer
# from scipy.spatial.distance import cosine

# class SentenceSimilarity:
#     def __init__(self, model_name):
#         self.model = SentenceTransformer(model_name)
    
#     def calculate_similarity(self, sentence1, sentence2):
#         embeddings1 = self.model.encode(sentence1, convert_to_tensor=True)
#         embeddings2 = self.model.encode(sentence2, convert_to_tensor=True)
#         similarity_score = 1 - cosine(embeddings1, embeddings2)
#         return similarity_score

# # Example usage
# # sentence1 = "Object oriented programming is a computer science concept."
# # sentence2 = "obama is the president of the united states."
# sentence1 = "John ate chicken"
# sentence2 = "chicken ate a john"

# model_name = 'bert-base-nli-mean-tokens'  # BERT model for sentence embeddings
# similarity_obj = SentenceSimilarity(model_name)
# similarity_score = similarity_obj.calculate_similarity(sentence1, sentence2)
# print("Similarity score:", similarity_score)



## transformers

# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# class SentenceSimilarity:
    
#     def __init__(self, model_name="bert-base-uncased"):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
#     def calculate_similarity(self, sentence1, sentence2):
#         # Tokenize sentences
#         inputs = self.tokenizer(sentence1, sentence2, return_tensors="pt", padding=True, truncation=True)
        
#         # Calculate similarity score
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             logits = outputs.logits
#             probability = torch.softmax(logits, dim=1)
        
#         # Probability of label 1 indicates similarity score
#         similarity_score = probability[:, 1].item()
#         return similarity_score

# # Example usage
# sentence1 = "John ate chicken"
# sentence2 = "chicken ate a john"

# similarity_obj = SentenceSimilarity()
# similarity_score = similarity_obj.calculate_similarity(sentence1, sentence2)
# print("Similarity score:", similarity_score)


# # # chatgpt api
# import os
# import pdb
# import re
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()
# # Set up OpenAI API key

# client = OpenAI(
#         api_key=os.environ.get("OPENAI_API_KEY"),
#     )
# # OpenAI.api_key = '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'

# def evaluate_similarity(reference_answer, candidate_answer):
#     # Concatenate the reference and candidate answers with a separator
#     # prompt = f"We are assessing candidates' understanding of technical concepts based on their responses in an interview. Below is the original answer we consider as a standard for understanding a specific technical question. Following that, you'll find a candidate's response to the same question. Your task is to analyze the candidate's response in comparison to the original answer and evaluate the similarity in context and understanding. Provide a summary that includes whether the candidate has grasped the key concepts, the depth of their understanding, and any significant discrepancies or additional insights they've offered compared to the original answer. Reference answer: {reference_answer}\nCandidate answer: {candidate_answer}\n"
#     prompt = f"We are assessing candidates' understanding of technical concepts based on their responses in an interview. Below is the original answer we consider as a standard for understanding a specific technical question. Following that, you'll find a candidate's response to the same question. Your task is to analyze the candidate's response in comparison to the original answer and evaluate the similarity in context and understanding. Provide a summary that includes whether the candidate has grasped the key concepts, the depth of their understanding, and any significant discrepancies or additional insights they've offered compared to the original answer. Also, Give a confidence/similarity numerical score comparing the reference answer with Candidate's answer ranging from approximately 0 to 1. Store the decimal score in variable ""Score"". Reference answer: {reference_answer}\nCandidate answer: {candidate_answer}\n"
#     # Use the Davinci model for text completion
#     response = client.completions.create(
#         model="gpt-3.5-turbo-instruct",
#         prompt=prompt,
#         max_tokens=3000,
#         temperature=0.5,
        
#     )

#     # Extract the confidence score from the completion
#     # pdb.set_trace()
#     confidence_score = response.choices[0]
#     print(confidence_score)

#     return confidence_score
#     # confidence_score = response.choices[0].logprobs.token_logprobs[-1]["logit"]
    
#     # # Normalize the confidence score to range from 0 to 1
#     # normalized_confidence_score = 1 / (1 + 2.71828 ** (-confidence_score))

#     # return normalized_confidence_score

# # Example usage
# reference_answer = "OOP is a programming paradigm based on the concept of 'objects', which can contain data in the form of fields (often known as attributes or properties), and code in the form of procedures (often known as methods). A feature of objects is an object's procedures that can access and often modify the data fields of the object with which they are associated (objects have a notion of 'this' or 'self'). In OOP, computer programs are designed by making them out of objects that interact with one another. OOP languages are diverse, but the most popular ones are class-based, meaning that objects are instances of classes, which also determine their types."
# # candidate_answer = "OOP is computer networking concept used for network transmission"
# candidate_answer = "Object-oriented programming is a programming paradigm based on the concept of objects, which can contain data and code: data in the form of fields, and code in the form of procedures."
# confidence_score = evaluate_similarity(reference_answer, candidate_answer)
# match = re.search(r'Score: (\d+\.\d+)',  confidence_score.text)
# if match:
#     confidence_score = float(match.group(1))
#     print("Confidence Score:", confidence_score)
# else:
#     print("Error: Confidence score not found")
# print("Confidence Score:", confidence_score)


from flask import Flask, request, jsonify
import re
from openai import OpenAI
import os
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

# Set up OpenAI API key
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def evaluate_similarity(reference_answer, candidate_answer):
    prompt = f"We are assessing candidates' understanding of technical concepts based on their responses in an interview. Below is the original answer we consider as a standard for understanding a specific technical question. Following that, you'll find a candidate's response to the same question. Your task is to analyze the candidate's response in comparison to the original answer and evaluate the similarity in context and understanding. Provide a summary that includes whether the candidate has grasped the key concepts, the depth of their understanding, and any significant discrepancies or additional insights they've offered compared to the original answer. Also, Give a confidence/similarity numerical score comparing the reference answer with Candidate's answer ranging from approximately 0 to 1. Store the decimal score in variable 'Score'. Reference answer: {reference_answer}\nCandidate answer: {candidate_answer}\n"
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=3000,
        temperature=0.5,
    )

    completion_text = response.choices[0].text
    
    return completion_text

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:

        data = request.get_json()
        reference_answer = data['reference_answer']
        candidate_answer = data['candidate_answer']

        completion_text = evaluate_similarity(reference_answer, candidate_answer)

        full_answer = completion_text

        match = re.search(r'Score: (\d+\.\d+)', completion_text)
        if match:
            # confidence_score = float(match.group(1))
            # return jsonify({'confidence_score': confidence_score, 'feedback': full_answer})
            return jsonify({'feedback': full_answer})
        else:
            return jsonify({'error': 'Confidence score not found'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500    

if __name__ == '__main__':
    app.run(debug=True)