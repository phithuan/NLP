import tkinter as tk
from tkinter import ttk
from collections import Counter
import nltk
from nltk.util import ngrams

# Tải dữ liệu token hóa nếu chưa có
nltk.download('punkt')

# ========== XỬ LÝ DỮ LIỆU NGÔN NGỮ ==========

# Tiền xử lý văn bản: chuyển chữ thường và tách từ
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return tokens

# Tạo mô hình n-gram
def build_ngram_model(tokens, n):
    n_grams = list(ngrams(tokens, n))
    return Counter(n_grams)

# Dự đoán từ tiếp theo dựa trên tiền tố
def predict_next_word(model, prefix):
    prefix = tuple(prefix)
    candidates = {k[-1]: v for k, v in model.items() if k[:-1] == prefix}
    return sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:5]

# ========== HÀM SỰ KIỆN GUI ==========

# Cập nhật danh sách gợi ý khi người dùng gõ
def update_suggestions(event=None):
    words = entry_var.get().strip().split()
    if not words or len(words) < n - 1:
        suggestion_list.delete(0, tk.END)
        return

    prefix = words[-(n - 1):]
    suggestions = predict_next_word(ngram_model, prefix)

    suggestion_list.delete(0, tk.END)
    for word, _ in suggestions:
        suggestion_list.insert(tk.END, word)

    if suggestion_list.size() > 0:
        suggestion_list.selection_set(0)
        suggestion_list.activate(0)

# Gắn từ đã chọn khi nhấn Tab
def on_tab(event):
    if suggestion_list.size() == 0:
        return

    try:
        index = suggestion_list.curselection()[0]
    except IndexError:
        index = 0

    selected_word = suggestion_list.get(index)
    current_text = entry_var.get().strip()
    new_text = f"{current_text} {selected_word}"
    entry_var.set(new_text)
    entry.icursor(tk.END)
    suggestion_list.delete(0, tk.END)
    return "break"

# Xử lý khi nhấn ↑ hoặc ↓
def on_arrow_key(event):
    if suggestion_list.size() == 0:
        return "break"  # Ngăn Tkinter xử lý mặc định

    try:
        index = suggestion_list.curselection()[0]
    except IndexError:
        index = -1  # Nếu chưa chọn dòng nào

    if event.keysym == "Down":
        new_index = index + 1 if index < suggestion_list.size() - 1 else index
    elif event.keysym == "Up":
        new_index = index - 1 if index > 0 else index
    else:
        return "break"

    suggestion_list.selection_clear(0, tk.END)
    suggestion_list.selection_set(new_index)
    suggestion_list.activate(new_index)
    suggestion_list.see(new_index)  # Tự cuộn nếu cần
    return "break"  # Ngăn Entry xử lý phím ↑↓

# Xử lý khi click đúp vào từ gợi ý
def on_click_suggestion(event):
    try:
        index = suggestion_list.curselection()[0]
        selected_word = suggestion_list.get(index)
        current_text = entry_var.get().strip()
        new_text = f"{current_text} {selected_word}"
        entry_var.set(new_text)
        entry.icursor(tk.END)
        suggestion_list.delete(0, tk.END)
    except IndexError:
        pass

# ========== ĐỌC DỮ LIỆU VÀ TẠO MÔ HÌNH ==========

# Đọc file dữ liệu huấn luyện
with open("data_ctut.txt", "r", encoding="utf-8") as file:
    text = file.read()

tokens = preprocess_text(text)
n = 2  # Sử dụng bigram, có thể đổi thành 3 nếu muốn dùng trigram
ngram_model = build_ngram_model(tokens, n)

# ========== GIAO DIỆN TKINTER ==========

root = tk.Tk()
root.title("Gợi ý từ thông minh")
root.geometry("600x300")
root.resizable(False, False)

# Entry nhập văn bản
entry_var = tk.StringVar()
entry = tk.Entry(root, textvariable=entry_var, font=("Arial", 14), width=50)
entry.pack(pady=10)

# Khung chứa listbox + scrollbar
frame = tk.Frame(root)
frame.pack()

scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
suggestion_list = tk.Listbox(
    frame, font=("Arial", 14), width=50, height=6, yscrollcommand=scrollbar.set
)
scrollbar.config(command=suggestion_list.yview)

suggestion_list.pack(side=tk.LEFT, fill=tk.BOTH)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# ========== GẮN SỰ KIỆN ==========

entry.bind("<KeyRelease>", update_suggestions)         # Gợi ý khi gõ
entry.bind("<Tab>", on_tab)                            # Gợi ý khi Tab
entry.bind("<Up>", on_arrow_key)                       # Di chuyển ↑
entry.bind("<Down>", on_arrow_key)                     # Di chuyển ↓
suggestion_list.bind("<Double-Button-1>", on_click_suggestion)  # Click gợi ý

# Chạy giao diện
root.mainloop()
