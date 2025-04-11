import nltk
import collections
import random
import pickle
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tải dữ liệu cần thiết từ NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Khởi tạo stopwords và lemmatizer
stop_word = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Hàm đọc nội dung các file văn bản từ thư mục
def load_file(directory):
    result = []
    for fname in os.listdir(directory):
        path = os.path.join(directory, fname)
        if os.path.isfile(path):
            with open(path, 'r', encoding='ISO-8859-1') as f:
                result.append(f.read())
    return result

# Hàm tiền xử lý một câu
def preprocess_sentence(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    word_counts = collections.Counter(tokens)
    uncommon_words = [word for word, count in word_counts.most_common()[:-10:-1]]
    tokens = [w for w in tokens if w.isalpha() and w not in stop_word and w not in uncommon_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens

# Hàm trích xuất đặc trưng
def feature_extraction(tokens):
    return dict(collections.Counter(tokens))

# Hàm chia train/test
def train_test_split(dataset, train_size=0.7):
    split = int(len(dataset) * train_size)
    return dataset[:split], dataset[split:]

# ==== Bắt đầu huấn luyện mô hình ====

# Đọc dữ liệu Enron
positive_examples = load_file('data/spam')  # Label = 1
negative_examples = load_file('data/ham')   # Label = 0

# Tiền xử lý
positive_examples = [preprocess_sentence(email) for email in positive_examples]
negative_examples = [preprocess_sentence(email) for email in negative_examples]

# Gắn nhãn
positive_examples = [(email, 1) for email in positive_examples]
negative_examples = [(email, 0) for email in negative_examples]

# Gộp và shuffle
all_examples = positive_examples + negative_examples
random.shuffle(all_examples)

print(f"{len(all_examples)} emails processed.")

# Trích xuất đặc trưng
featurized = [(feature_extraction(email), label) for email, label in all_examples]

# Tách train/test
training_set, test_set = train_test_split(featurized)

# Huấn luyện mô hình Naive Bayes
model = nltk.classify.NaiveBayesClassifier.train(training_set)

# Đánh giá
train_acc = nltk.classify.accuracy(model, training_set)
test_acc = nltk.classify.accuracy(model, test_set)

print(f"✅ Accuracy on training set: {train_acc * 100:.2f}%")
print(f"✅ Accuracy on test set: {test_acc * 100:.2f}%")

# Lưu model vào file .pkl để dùng cho web
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Mô hình đã được lưu thành công vào file: model.pkl")
