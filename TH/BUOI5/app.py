from flask import Flask, render_template, request
import nltk
import collections
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import pickle

# === Tải dữ liệu NLTK (nếu chưa tải) ===
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# === Khởi tạo Flask app ===
app = Flask(__name__)

# === Khởi tạo stop words và lemmatizer ===
stop_word = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# === Hàm tiền xử lý ===
def preprocess_sentence(sentence):
    tokens = word_tokenize(sentence.lower())
    word_counts = collections.Counter(tokens)
    uncommon_words = [word for word, count in word_counts.most_common()[:-10:-1]]
    tokens = [w for w in tokens if w.isalpha() and w not in stop_word and w not in uncommon_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens

# === Hàm trích xuất đặc trưng ===
def feature_extraction(tokens):
    return dict(collections.Counter(tokens))

# === Load mô hình đã huấn luyện ===
# Lưu mô hình bằng pickle sau khi huấn luyện: pickle.dump(model, open('model.pkl', 'wb'))
# Load lại để sử dụng
model = pickle.load(open('model.pkl', 'rb'))  # Đảm bảo bạn đã có file model.pkl trước đó

# === Route trang chính ===
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        email = request.form['email']
        tokens = preprocess_sentence(email)
        features = feature_extraction(tokens)
        label = model.classify(features)
        result = 'SPAM' if label == 1 else 'HAM (Không spam)'
    return render_template('index.html', result=result)

# === Chạy app local ===
if __name__ == '__main__':
    app.run(debug=True)
