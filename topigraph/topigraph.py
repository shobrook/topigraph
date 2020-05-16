# Third Party
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from numpy import dot
from numpy.linalg import norm
from youtube_transcript_api import YouTubeTranscriptApi
from wikipedia import search as wikisearch
from graphlou import detect_communities

# Standard Library
import heapq
from collections import defaultdict


def fetch_transcript(video_id="RD-9Ghvt480", max_num_lines=None):
    data = YouTubeTranscriptApi.get_transcript(video_id)
    num_lines = 0
    for line in data:
        if max_num_lines and num_lines >= max_num_lines:
            break

        num_lines += 1
        yield line["text"]


def preprocess_sentences(transcript):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    for sentence in transcript:
        word_tokens = (w.lower() for w in word_tokenize(sentence))
        word_tokens = (w for w in word_tokens if w not in stop_words)
        word_tokens = ((lemmatizer.lemmatize(w), w) for w in word_tokens)

        yield word_tokens


def vectorize_sentences(pp_transcript, max_vec_size=25):
    dataset = []
    word2count = defaultdict(lambda: 0)
    for sentence in pp_transcript:
        sentence_for_dataset = []
        for (lemma, word) in sentence:
            word2count[lemma] += 1
            sentence_for_dataset.append(word)

        dataset.append(sentence_for_dataset)

    freq_words = heapq.nlargest(max_vec_size, word2count, key=word2count.get)
    sent_vectors = []
    for sentence in dataset:
        sent_vector = [sentence.count(w) for w in freq_words]
        sent_vectors.append(sent_vector)

    return sent_vectors, dataset


def create_similarity_matrix(sent_vectors):
    sim_matrix = []
    for i, source in enumerate(sent_vectors):
        row = []
        for j, target in enumerate(sent_vectors):
            # Keeps matrix left-triangular
            if j > i:
                break

            # TODO: Use something better than cosine similarity
            numer = dot(source, target)
            denom = norm(source) * norm(target)
            cosine_sim = numer / denom if denom else 0.0
            row.append(cosine_sim)

        sim_matrix.append(row)

    return sim_matrix


def extract_topic_labels(clusters, node2sent):
    # valid_tags = ("NN", "NNS", "NNP", "NNPS")
    valid_tags = ("NNP")
    topic_labels = []
    for cluster in clusters:
        words = [word for node in cluster for word in node2sent[node]]
        keywords = [word for word, pos in pos_tag(words) if pos in valid_tags]
        keywords = keywords[:10] # TODO: Rank keywords

        if not keywords:
            continue

        topic_labels.append(wikisearch(" ".join(keywords))[0])

    return topic_labels


if __name__ == "__main__":
    print("\tFetching transcript")

    transcript = fetch_transcript(max_num_lines=200)

    print("\tPreprocessing transcript")

    pp_transcript = preprocess_sentences(transcript)

    print("\tVectorizing sentences in transcript")

    sent_vectors, node2sent = vectorize_sentences(pp_transcript)

    print("\tCreating similarity matrix")

    sim_matrix = create_similarity_matrix(sent_vectors)

    print("\tClustering sentences with Louvain's algorithm (takes an eternity)")

    clusters = detect_communities(sim_matrix)

    print("\tExtracting topic labels")

    topics = extract_topic_labels(clusters, node2sent)

    print(topics)
